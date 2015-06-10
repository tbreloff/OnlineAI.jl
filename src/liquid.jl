

# NOTE: consider this experimental at best... you probably shouldn't use it


# ---------------------------------------------------------------------

# we use a vector of gaussian receptive fields to map continuous real-valued inputs
# to a series of spike trains

type GaussianReceptiveField
	variance::Variance
	grf_offset::Float64
	grf_width_factor::Float64
end


# create a field that covers a specific area (partially overlapping with the next closest one)
const grf_offsets = [0.0, 2.0, 6.0, 14.0]
const grf_width_factors = [0.5, 1.0, 2.0, 4.0] * 1.5
function GaussianReceptiveField(variance::Variance, fieldnum::Int)
	idx = floor(Int, fieldnum/2) + 1
	@assert idx > 0 && idx <= length(grf_offsets)
	GaussianReceptiveField(variance, (iseven(fieldnum) ? 1.0 : -1.0) * grf_offsets[idx], grf_width_factors[idx])
end


const STANDARD_NORMAL = Normal()
const CENTER_PDF = pdf(STANDARD_NORMAL, 0.0)
function value(grf::GaussianReceptiveField, x::Float64)
	pdf(STANDARD_NORMAL, (standardize(grf.variance, x) + grf.grf_offset) / grf.grf_width_factor) / CENTER_PDF
end

OnlineStats.update!(grf::GaussianReceptiveField, x::Float64) = update!(grf.variance, x)


function Qwt.plot(grfs::Vector{GaussianReceptiveField}, rng::FloatIterable)
	y = Float64[value(grfs[i],r) for r in rng, i in 1:length(grfs)]
	plot(rng,y)
end



# ---------------------------------------------------------------------

# When a neuron nᵢ fires, it sends a pulse of wᵢⱼ to the dᵗʰ position of the circular buffer of future pulses for nⱼ.
# In other words: at time t there is a spike in neuron nᵢ.  at time t+d we apply a pulse wᵢⱼ to neuron nⱼ

abstract Synapse
abstract SpikingNeuron <: Node

# connects neurons together
type DiscreteSynapse
	postsynapticNeuron::SpikingNeuron
	weight::Float64
	delay::Int # number of periods to delay the pulse
end

function fire!(synapse::DiscreteSynapse)
	postsynapticNeuron.futurePulses[synapse.delay] += synapse.weight
end

# ---------------------------------------------------------------------

# we use the model [ uₜ = pulsesₜ + (1-λ) * uₜ₋₁ ] to update
type DiscreteLeakyIntegrateAndFireNeuron <: SpikingNeuron
	position::NTuple{Int,3}  # position in neuronal column
	excitatory::Bool
	futurePulses::CircularBuffer{Float64}  # future pulses by time offset {pulseₜ₊₁, pulseₜ₊₂, ...}
	u::Float64		  		# current state
	# uᵣₑₛₜ::Float64  # resting state # note: assume 0 for now
	ϑ::Float64  		# threshold level
	λ::Float64 				# exponential decay rate
	refractoryPeriodsRemaining::Int  # no activity allowed for this many periods after a spike
	refractoryPeriodsTotal::Int  # no activity allowed for this many periods after a spike
	fired::Bool  # did the neuron fire in the most recent period?
	synapses::Vector{Synapse}
end

const MAX_FUTURE = 10
const DEFAULT_THRESHOLD = 2.0
const DEFAULT_REFRACTORY_PERIOD = 4

function DiscreteLeakyIntegrateAndFireNeuron(position::NTuple{Int,3}, excitatory::Bool, λ::Float64)
	DiscreteLeakyIntegrateAndFireNeuron(position,
																			excitatory,
																			CircularBuffer(Float64, MAX_FUTURE, 0.0),
																			0.0,
																			DEFAULT_THRESHOLD,
																			λ,
																			0,
																			DEFAULT_REFRACTORY_PERIOD,
																			false,
																			Synapse[])
end

# stepping through time involves 2 actions:
# - incorporate pulses into u (spike --> reset and apply pulse to other neuron's futurePulses) and decay
# - push! 0 onto futurePulses to step forward into time

function OnlineStats.update!(neuron::DiscreteLeakyIntegrateAndFireNeuron)

	# if we're in the refractory period, don't adjust u
	if neuron.refractoryPeriodsRemaining > 0
		neuron.refractoryPeriodsRemaining -= 1
	else
		neuron.u *= (1.0 - neuron.λ)
		neuron.u += futurePulses[1]
	end

	# step to next time period
	push!(neuron.futurePulses, 0.0)
end

# fire/reset neurons that have crossed the threshold
function fire!(neuron::DiscreteLeakyIntegrateAndFireNeuron)
	neuron.fired = neuron.u >= neuron.ϑ
	if neuron.fired
		foreach(synapses, fire!)
		neuron.u = 0.0
		neuron.refractoryPeriodsRemaining = neuron.refractoryPeriodsTotal
	end
end


# ---------------------------------------------------------------------

# give default values of C from the P(connection) = C * exp(-(D(a,b)/λ)^2)

const C_EE = 0.3
const C_EI = 0.2
const C_IE = 0.4
const C_II = 0.1

function C(a_excitatory::Bool, b_excitatory::Bool)
	if a_excitatory
		return b_excitatory ? C_EE : C_EI
	else
		return b_excitatory ? C_IE : C_II
	end
end

# ---------------------------------------------------------------------

typealias Neurons Vector{DiscreteLeakyIntegrateAndFireNeuron}

type Liquid
	neurons::Neurons
	inputNeurons::Neurons
	outputNeurons::Neurons
end

function Liquid(l::Int, w::Int, h::Int; 
				pctInhibitory::Float64 = 0.2,
				λ::Float64 = 0.01,
				pctInput::Float64 = 0.2,
				pctOutput::Float64 = 0.4)
	
	neurons = vec([DiscreteLeakyIntegrateAndFireNeuron((i, j, k), rand() > pctInhibitory, λ) for i in 1:l, j in 1:w, k in 1:h])
	inputNeurons = sample(neurons, round(Int, pctInput * length(neurons)))
	outputNeurons = sample(neurons, round(Int, pctOutput * length(neurons)))
	Liquid(neurons, inputNeurons, outputNeurons)
end

OnlineStats.update!(liquid::Liquid) = foreach(liquid.neurons, update!, fire!)

# ---------------------------------------------------------------------


# maintains a list of GRFs which correspond to several spike trains for each input value
type LiquidInput
	K::Int # number of inputs
	M::Int # number of receptive fields per input
	variances::Vector{Variance}
	grfs::Matrix{GaussianReceptiveField}  # K x M matrix of GRFs
	synapses::Matrix{Synapse}   # K x M matrix of synapses to input neurons
end

const GRF_MULTS = map(abs2, 1:10)[2:end]

function createGRF(variance::Variance, j::Int, M::Int)
	n = trunc(Int, M/2)
	idx = (j-2) % n + 1
	if idx == 0
		return GaussianReceptiveField(dist)
	else
		σ = dist.σ * GRF_MULTS[idx]
		μ = 1.5 * (idx <= n + 1 ? -1.0 : 1.0) * σ + dist.μ
		return GaussianReceptiveField(Normal(μ, σ))
	end
end


function LiquidInput(variances::Vector{Variance}, liquid::Liquid)
	K = length(variances)
	M = 5  # TODO: make variable?
	grfs = GaussianReceptiveField[createGRF(variances[i], j, M) for i in 1:K, j in 1:M]
	
	# TODO: create synapses!

	LiquidInput(K, M, variances, grfs, synapses)
end

OnlineStats.update!(liquidInput::LiquidInput, x::VecF) = foreach(liquidInput.variances, update!)

# ---------------------------------------------------------------------

type LiquidStateMachine
	# TODO: holds LiquidInput, Liquid, LiquidOutput, OutputModel
end

function OnlineStats.update!(lsm::LiquidStateMachine, x::VecF)
	# TODO:
	# update LiquidInput
end

function StatsBase.predict(lsm::LiquidStateMachine)
	# TODO: given the current liquid state and readout model, predict the future
end

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
