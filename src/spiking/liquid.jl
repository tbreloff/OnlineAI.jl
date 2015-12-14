
# NOTE: consider this experimental at best... you probably shouldn't use it

# ---------------------------------------------------------------------

type LiquidParams
  l::Int  # column dimensions l x w x h
  w::Int
  h::Int
  neuronType::DataType
  pctInhibitory::Float64
  decayRateDist::Distribution{Univariate,Continuous}
  λ::Float64  # used in probability of synapse connection
  pctInput::Float64
  pctOutput::Float64
  readout::Readout
  dt::Float64         # ms - simulation time step
  # τu::Float64         # ms - membrane potential decay period
  # τq::Float64          # ms - pulse decay period
  uDecayRate::Float64    # 1/ms - equal to 1/τu = membrane potential decay rate
  qDecayRate::Float64    # 1/ms - equal to 1/τq = pulse decay rate
  urest::Float64       # mV - resting membrane potential
  ufire::Float64      # mV - membrane potential immediately after firing a spike
  baseThreshold::Float64
end

function LiquidParams(; l::Int = 3,
                        w::Int = l,
                        h::Int = 5,
                        neuronType::DataType = SRMNeuron,
                        pctInhibitory::Float64 = 0.2,
                        decayRateDist::Distribution{Univariate,Continuous} = Uniform(0.8, 0.99),
                        λ::Float64 = 1.0,
                        pctInput::Float64 = 0.1,
                        pctOutput::Float64 = 0.4,
                        readout::Readout = FireReadout(),
                        dt::Float64 = 0.25,
                        # τu::Float64 = 25.0,
                        # τq::Float64 = 4.0,
                        uDecayRate::Float64 = 1.0 / 25.0,
                        qDecayRate::Float64 = 1.0 / 4.0,
                        urest::Float64 = 0.0,
                        ufire::Float64 = -0.4,
                        baseThreshold::Float64 = 2.0)
  LiquidParams(l, w, h, neuronType, pctInhibitory, decayRateDist, λ, pctInput, pctOutput, readout, dt, uDecayRate, qDecayRate, urest, ufire, baseThreshold)
end


Base.print(io::IO, l::LiquidParams) = print(io, "Params{$(l.l)x$(l.w)x$(l.h), pcti=$(l.pctInhibitory), λ=$(l.λ), pctout=$(l.pctOutput)}")
Base.show(io::IO, l::LiquidParams) = print(io, l)


# ---------------------------------------------------------------------

# maintains a list of GRFs which correspond to several spike trains for each input value
type LiquidInputs{T <: LiquidInput}
  inputs::Vector{T}   # one per input
end


function OnlineStats.update!(inputs::LiquidInputs, x::VecF, dt::Float64)
  for i in 1:length(inputs.inputs)
    update!(inputs.inputs[i], x[i], dt)
  end
end

# ---------------------------------------------------------------------

# methods to compute probability of synaptic connection
#   P(connection) = C * exp(-(D(a,b)/λ)^2)

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
const UNIF_WEIGHT = Uniform(0.1, 1.0)

function weight(n::SpikingNeuron)
  (n.excitatory ? 1.0 : -1.0) * rand(UNIF_WEIGHT)
end

bound{T<:Real}(x::T, l::T, u::T) = max(l, min(x, u))

function delay(n::SpikingNeuron)
  sample(1:MAX_FUTURE)
end

# ---------------------------------------------------------------------

type Liquid{T}
  params::LiquidParams
  neurons::Vector{T}
  outputNeurons::Vector{T}
end

Base.print(io::IO, l::Liquid) = print(io, "Liquid{n=$(length(l.neurons)), nout=$(length(l.outputNeurons))}")
Base.show(io::IO, l::Liquid) = print(io, l)

function Liquid{T}(::Type{T}, params::LiquidParams) 

  # create neurons in an (w x w x h) column
  # neurons = DiscreteLeakyIntegrateAndFireNeuron[]
  neurons = T[]
  for i in 1:params.l
    for j in 1:params.w
      for k in 1:params.h
        excitatory = rand() > params.pctInhibitory
        neuron = T([i,j,k], excitatory, params)
        push!(neurons, neuron)
      end
    end
  end

  # output from a random subset of the neuronal column
  outputNeurons = sample(neurons, params.pctOutput)

  # randomly connect the neurons
  for n1 in neurons
    for n2 in neurons
      if rand() <= probabilityOfConnection(n1, n2, params.λ)
        # synapse = DelaySynapse(n2, weight(n1), delay(n1))
        synapse = SRMSynapse(n2, weight(n1))
        push!(n1.synapses, synapse)
      end
    end
    println(n1, ", Synapses:")
    for s in n1.synapses println("     ", s) end
  end

  Liquid{T}(params, neurons, outputNeurons)
end

StatsBase.sample{T<:SpikingNeuron}(neurons::Vector{T}, pct::Float64) = sample(neurons, round(Int, pct * length(neurons)))

function OnlineStats.update!(liquid::Liquid, dt::Float64)
  for neuron in liquid.neurons
    update!(neuron, dt)
  end
  foreach(liquid.neurons, fire!)
end


# ---------------------------------------------------------------------

# manages the various layers and flow: input --> liquid --> readout --> readout model

type LiquidStateMachine <: OnlineStat
  nin::Int
  nout::Int
  liquid::Liquid
  inputs::LiquidInputs
  readout::Readout
  readoutModels::Vector{OnlineStat}
  n::Int
end

function LiquidStateMachine(params::LiquidParams, numInputs::Int, numOutputs::Int)
  # initialize liquid
  liquid = Liquid(params.neuronType, params)

  # create input structure
  wgt = ExponentialWeighting(10000)
  variances = [Variance(wgt) for i in 1:numInputs]
  inputs = LiquidInputs(liquid, variances)

  # create readout models
  # TODO: make readout model more flexible... param
  # readoutModels = OnlineStat[OnlineFLS(length(liquid.outputNeurons), 0.00001, wgt) for i in 1:numOutputs]
  readoutModels = OnlineStat[Adagrad(length(liquid.outputNeurons); penalty = L2Penalty(0.00001)) for i in 1:numOutputs]

  LiquidStateMachine(numInputs, numOutputs, liquid, inputs, params.readout, readoutModels, 0)
end

# liquidState(lsm::LiquidStateMachine) = Float64[float(neuron.fired) for neuron in lsm.liquid.outputNeurons]
OnlineStats.statenames(lsm::LiquidStateMachine) = [:liquidState, :nobs]
OnlineStats.state(lsm::LiquidStateMachine) = Any[liquidState(lsm), nobs(lsm)]


function OnlineStats.update!(lsm::LiquidStateMachine, x::VecF, y::VecF, dt::Float64 = lsm.liquid.params.dt)
  update!(lsm.inputs, x, dt)   # update input neurons
  update!(lsm.liquid, dt)     # update liquid state

  # update readout models
  # TODO: liquidState should be more flexible... multiple models, recent window averages, etc
  update!(lsm.readout, lsm.liquid.outputNeurons)
  state = liquidState(lsm, lsm.readout)
  for (i,model) in enumerate(lsm.readoutModels)
    update!(model, state, y[i])
  end

  lsm.n += 1
end

# given the current liquid state and readout model, predict the future
function StatsBase.predict(lsm::LiquidStateMachine)
  state = liquidState(lsm, lsm.readout)
  Float64[predict(model, state) for model in lsm.readoutModels]
end

# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
