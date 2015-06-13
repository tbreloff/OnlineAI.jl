

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
const grf_offsets = [0.0, 2.0, 6.0, 14.0] / 2.0
const grf_width_factors = [0.5, 1.0, 2.0, 4.0] * 0.5
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
	synapse.postsynapticNeuron.futurePulses[synapse.delay] += synapse.weight
end

Base.print(io::IO, s::DiscreteSynapse) = print(io, "DiscreteSynapse{post=$(s.postsynapticNeuron), wgt=$(s.weight), delay=$(s.delay)}")
Base.show(io::IO, s::DiscreteSynapse) = print(io, s)

# ---------------------------------------------------------------------

# we use the model [ uₜ = pulsesₜ + (1-λ) * uₜ₋₁ ] to update
type DiscreteLeakyIntegrateAndFireNeuron <: SpikingNeuron
	position::VecI  # 3x1 position in neuronal column
	excitatory::Bool
	futurePulses::CircularBuffer{Float64}  # future pulses by time offset {pulseₜ₊₁, pulseₜ₊₂, ...}
	u::Float64		  		# current state
	# uᵣₑₛₜ::Float64  # resting state # note: assume 0 for now
	ϑ::Float64  		# threshold level
	decayRate::Float64 				# exponential decay rate
	refractoryPeriodsRemaining::Int  # no activity allowed for this many periods after a spike
	refractoryPeriodsTotal::Int  # no activity allowed for this many periods after a spike
	fired::Bool  # did the neuron fire in the most recent period?
	synapses::Vector{DiscreteSynapse}
end

# TODO: make these parameters
const MAX_FUTURE = 3
const DEFAULT_THRESHOLD = 1.0
const DEFAULT_REFRACTORY_PERIOD = 1

function DiscreteLeakyIntegrateAndFireNeuron(position::VecI, excitatory::Bool, decayRate::Float64)
	DiscreteLeakyIntegrateAndFireNeuron(position,
																			excitatory,
																			CircularBuffer(Float64, MAX_FUTURE, 0.0),
																			0.0,
																			DEFAULT_THRESHOLD,
																			decayRate,
																			0,
																			DEFAULT_REFRACTORY_PERIOD,
																			false,
																			DiscreteSynapse[])
end

Base.print(io::IO, n::DiscreteLeakyIntegrateAndFireNeuron) = 
	print(io, "DLIFNeuron{pos=$(n.position), exite=$(n.excitatory), u=$(n.u), fired=$(n.fired), nsyn=$(length(n.synapses))}")
Base.show(io::IO, n::DiscreteLeakyIntegrateAndFireNeuron) = print(io, n)

# stepping through time involves 2 actions:
# - incorporate pulses into u (spike --> reset and apply pulse to other neuron's futurePulses) and decay
# - push! 0 onto futurePulses to step forward into time

function OnlineStats.update!(neuron::DiscreteLeakyIntegrateAndFireNeuron)

	# if we're in the refractory period, don't adjust u
	if neuron.refractoryPeriodsRemaining > 0
		neuron.refractoryPeriodsRemaining -= 1
	else
		neuron.u *= neuron.decayRate
		neuron.u += neuron.futurePulses[1]
	end

	# don't let it go negative
	neuron.u = max(0.0, neuron.u)

	# step to next time period
	push!(neuron.futurePulses, 0.0)
end

# fire/reset neurons that have crossed the threshold
function fire!(neuron::DiscreteLeakyIntegrateAndFireNeuron)
	neuron.fired = neuron.u >= neuron.ϑ
	if neuron.fired
		foreach(neuron.synapses, fire!)
		neuron.u = 0.0
		neuron.refractoryPeriodsRemaining = neuron.refractoryPeriodsTotal
	end
end


function OnlineStats.state(neuron::DiscreteLeakyIntegrateAndFireNeuron)
	neuron.u / neuron.ϑ + 10.0 * float(neuron.fired)
end


# ---------------------------------------------------------------------

# methods to compute probability of synaptic connection
#		P(connection) = C * exp(-(D(a,b)/λ)^2)

const C_EE = 0.3
const C_EI = 0.2
const C_IE = 0.4
const C_II = 0.1
function C(n1::SpikingNeuron, n2::SpikingNeuron)
	n1.excitatory && (return n2.excitatory ? C_EE : C_EI)
	n2.excitatory ? C_IE : C_II
end

function distance(n1::SpikingNeuron, n2::SpikingNeuron)
	norm(n1.position - n2.position)
end

function probabilityOfConnection(n1::SpikingNeuron, n2::SpikingNeuron, λ::Float64)
	C(n1, n2) * exp(-((distance(n1, n2) / λ) ^ 2))
end

# TODO: make this a parameter
const UNIF_WEIGHT = Uniform(0.1, 0.5)

function weight(n::SpikingNeuron)
	(n.excitatory ? 1.0 : -1.0) * rand(UNIF_WEIGHT)
end

bound{T<:Real}(x::T, l::T, u::T) = max(l, min(x, u))

function delay(n::SpikingNeuron)
	sample(1:MAX_FUTURE)
end

# ---------------------------------------------------------------------

type LiquidParams
	l::Int  # column dimensions l x w x h
	w::Int
	h::Int
	pctInhibitory::Float64
	decayRateDist::Distribution{Univariate,Continuous}
	λ::Float64  # used in probability of synapse connection
	pctOutput::Float64
end

function LiquidParams(; l::Int = 3,
												w::Int = l,
												h::Int = 5,
												pctInhibitory::Float64 = 0.2,
												decayRateDist::Distribution{Univariate,Continuous} = Uniform(0.8, 0.99),
												λ::Float64 = 0.1,
												pctOutput::Float64 = 0.4)
	LiquidParams(l, w, h, pctInhibitory, decayRateDist, λ, pctOutput)
end


Base.print(io::IO, l::LiquidParams) = print(io, "Params{$(l.l)x$(l.w)x$(l.h), pcti=$(l.pctInhibitory), λ=$(l.λ), pctout=$(l.pctOutput)}")
Base.show(io::IO, l::LiquidParams) = print(io, l)

# ---------------------------------------------------------------------

type Liquid
	params::LiquidParams
	neurons::Vector
	outputNeurons::Vector
end

Base.print(io::IO, l::Liquid) = print(io, "Liquid{n=$(length(l.neurons)), nout=$(length(l.outputNeurons))}")
Base.show(io::IO, l::Liquid) = print(io, l)

# function Liquid{SN<:SpikingNeuron}(::Type{SN},
# 				w::Int, h::Int; 
# 				pctInhibitory::Float64 = 0.2,
# 				decayRate::Float64 = 0.99,  # TODO: make this variable/random
# 				λ::Float64 = 0.01,
# 				pctOutput::Float64 = 0.4)

function Liquid(params::LiquidParams)	

	# create neurons in an (w x w x h) column
	neurons = DiscreteLeakyIntegrateAndFireNeuron[]
	for i in 1:params.l
		for j in 1:params.w
			for k in 1:params.h
				excitatory = rand() > params.pctInhibitory
				decayRate = rand(params.decayRateDist)
				neuron = DiscreteLeakyIntegrateAndFireNeuron([i, j, k], excitatory, decayRate)
				push!(neurons, neuron)
			end
		end
	end

	# neurons = vec([DiscreteLeakyIntegrateAndFireNeuron([i, j, k], rand() > params.pctInhibitory, rand(params.decayRateDist)) for i in 1:params.l, j in 1:params.w, k in 1:params.h])

	# output from a random subset of the neuronal column
	outputNeurons = sample(neurons, round(Int, params.pctOutput * length(neurons)))

	# randomly connect the neurons
	for n1 in neurons
		for n2 in neurons
			if rand() <= probabilityOfConnection(n1, n2, params.λ)
				synapse = DiscreteSynapse(n2, weight(n1), delay(n1))
				push!(n1.synapses, synapse)
			end
		end
		println(n1, ", Synapses:")
		for s in n1.synapses println("     ", s) end
	end

	Liquid(params, neurons, outputNeurons)
end

OnlineStats.update!(liquid::Liquid) = foreach(liquid.neurons, update!, fire!)

# ---------------------------------------------------------------------

type GRFInput
	grf::GaussianReceptiveField
	synapse::DiscreteSynapse
end



# ---------------------------------------------------------------------

# maintains a list of GRFs which correspond to several spike trains for each input value
type LiquidInput
	K::Int # number of inputs
	M::Int # number of receptive fields per input
	variances::Vector{Variance}
	inputs::Matrix{GRFInput}  # K x M matrix of inputs
end

function LiquidInput{W}(variances::Vector{Variance{W}}, liquid::Liquid)
	K = length(variances)
	M = 4  # TODO: make variable?
	inputs = GRFInput[createInput(liquid, variance, j+1) for variance in variances, j in 1:M]
	LiquidInput(K, M, variances, inputs)
end

function createInput(liquid::Liquid, variance::Variance, j::Int)
	grf = GaussianReceptiveField(variance, j)
	postsynapticNeuron = sample(liquid.neurons)
	synapse = DiscreteSynapse(postsynapticNeuron, postsynapticNeuron.ϑ, 1)
	GRFInput(grf, synapse)
end



function OnlineStats.update!(liquidInput::LiquidInput, x::VecF)
	for (i,var) in enumerate(liquidInput.variances)
		update!(var, x[i])
	end

	# for each field, fire! with probability indicated by GRF
	for i in 1:liquidInput.K
		for j in 1:liquidInput.M
			input = liquidInput.inputs[i,j]
			probSpike = value(input.grf, x[i])
			if rand() <= probSpike
				fire!(input.synapse)
			end
		end
	end
end



# ---------------------------------------------------------------------



# ---------------------------------------------------------------------

# manages the various layers and flow: input --> liquid --> output --> readout model

type LiquidStateMachine <: OnlineStat
	liquid::Liquid
	input::LiquidInput
	readoutModels::Vector{OnlineStat}
	n::Int
end

function LiquidStateMachine(params::LiquidParams, numInputs::Int, numOutputs::Int)
	# initialize liquid
	liquid = Liquid(params)

	# create input structure
	wgt = ExponentialWeighting(1000)
	variances = [Variance(wgt) for i in 1:numInputs]
	input = LiquidInput(variances, liquid)

	# create readout models
	# TODO: make readout model more flexible... param
	readoutModels = OnlineStat[OnlineFLS(length(liquid.outputNeurons), 0.001, wgt) for i in 1:numOutputs]

	LiquidStateMachine(liquid, input, readoutModels, 0)
end

liquidState(lsm::LiquidStateMachine) = Float64[float(neuron.fired) for neuron in lsm.liquid.outputNeurons]
OnlineStats.statenames(lsm::LiquidStateMachine) = [:liquidState, :nobs]
OnlineStats.state(lsm::LiquidStateMachine) = Any[liquidState(lsm), nobs(lsm)]


function OnlineStats.update!(lsm::LiquidStateMachine, y::VecF, x::VecF)
	update!(lsm.input, x)   # update input neurons
	update!(lsm.liquid)			# update liquid state

	# update readout models
	# TODO: liquidState should be more flexible... multiple models, recent window averages, etc
	state = liquidState(lsm)
	for (i,model) in enumerate(lsm.readoutModels)
		update!(model, y[i], state)
	end

	lsm.n += 1
end

# given the current liquid state and readout model, predict the future
function StatsBase.predict(lsm::LiquidStateMachine)
	state = liquidState(lsm)
	Float64[predict(model, state) for model in lsm.readoutModels]
end

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
