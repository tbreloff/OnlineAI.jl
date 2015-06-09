

# NOTE: consider this experimental at best... you probably shouldn't use it

using Distributions
using QuickStructs
using OnlineStats

# ---------------------------------------------------------------------

# we use a vector of gaussian receptive fields to map continuous real-valued inputs
# to a series of spike trains

type GaussianReceptiveField
	dist::Normal
end
GaussianReceptiveField(μ::Float64, σ::Float64) = GaussianReceptiveField(Normal(μ, σ))

probabilityOfSpike(grf::GaussianReceptiveField, x::Float64) = pdf(grf.dist, x) / pdf(grf.dist, grf.dist.μ)



# ---------------------------------------------------------------------

# When a neuron nᵢ fires, it sends a pulse of wᵢⱼ to the dᵗʰ position of the circular buffer of future pulses for nⱼ.
# In other words: at time t there is a spike in neuron nᵢ.  at time t+d we apply a pulse wᵢⱼ to neuron nⱼ

# connects neurons together
type Synapse
	postsynapticNeuron::DiscreteLeakyIntegrateAndFireNeuron
	weight::Float64
	delay::Int # number of periods to delay the pulse
end

function fire!(synapse::Synapse)
	postsynapticNeuron.futurePulses[synapse.delay] += synapse.weight
end

# ---------------------------------------------------------------------

# we use the model [ uₜ = pulsesₜ + (1-λ) * uₜ₋₁ ] to update
type DiscreteLeakyIntegrateAndFireNeuron <: Node
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

# stepping through time involves 2 actions:
# - incorporate pulses into u (spike --> reset and apply pulse to other neuron's futurePulses) and decay
# - push! 0 onto futurePulses to step forward into time

function update!(neuron::DiscreteLeakyIntegrateAndFireNeuron)

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
		apply(fire!, synapses)
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

Liquid(l::Int, w::Int, h::Int; 
				pctInhibitory::Float64 = 0.2,
				λ::Float64 = 
				)

function update!(liquid::Liquid)
	apply(update!, liquid.neurons)
	apply(fire!, liquid.neurons)
end

# ---------------------------------------------------------------------


# maintains a list of GRFs which correspond to several spike trains for each input value
type LiquidInput
	K::Int # number of inputs
	M::Int # number of receptive fields per input
	grfs::Matrix{GaussianReceptiveField}  # K x M matrix of GRFs
	synapses::Matrix{Synapse}   # K x M matrix of synapses to input neurons
end

const GRF_MULTS = map(abs2, 1:10)[2:end]

function createGRF(dist::Normal, j::Int, M::Int)
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


function LiquidInput(input_distributions::Vector{Normal}, liquid::Liquid)
	K = length(input_distributions)
	M = 5  # TODO: make variable?
	grfs = GaussianReceptiveField[createGRF(input_distributions[i], j, M) for i in 1:K, j in 1:M]
	
	# TODO: create synapses!

	LiquidInput(K, M, grfs, synapses)
end


# ---------------------------------------------------------------------

type LiquidStateMachine
	# TODO: holds LiquidInput, Liquid, LiquidOutput, OutputModel
end

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
