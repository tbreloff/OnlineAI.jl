
# Leaky Integrate and Fire (LIF)

type LIFSynapse <: Synapse
  postsynapticNeuron::SpikingNeuron
  weight::Float64
end

function fire!(synapse::LIFSynapse)
  # TODO add current to postsynaptic neuron
end

Base.print(io::IO, s::LIFSynapse) = print(io, "LIFSynapse{post=$(s.postsynapticNeuron), wgt=$(s.weight)}")
Base.show(io::IO, s::LIFSynapse) = print(io, s)


# ---------------------------------------------------------------------

const τm = 30.0         # ms - membrane time constant
const urest = 0.0       # mV - resting membrane potential
const Rm = 1.0          # MΩ - input resistence
const I_inject = 13.5   # nA - current from injection
const I_noise = 1.0     # nA - current noise
const τe = 2.0          # ms - refractory period for exitatory neurons
const τi = 2.0          # ms - refractory period for inhibitory neurons

type LIFNeuron <: SpikingNeuron
  position::VecI
  excitatory::Bool
  u::Float64      # voltage (Vm)
  ϑ::Float64      # voltage threshold
  I_synapse::Float64  # current from synapse
  tf::Float64     # last firing time
  synapses::Vector{LIFSynapse}
end

LIFNeuron(pos::VecI, exitatory::Bool)


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

# Discrete LIF (my design... not biologically plausible)

# When a neuron nᵢ fires, it sends a pulse of wᵢⱼ to the dᵗʰ position of the circular buffer of future pulses for nⱼ.
# In other words: at time t there is a spike in neuron nᵢ.  at time t+d we apply a pulse wᵢⱼ to neuron nⱼ


# connects neurons together
type DelaySynapse <: Synapse
  postsynapticNeuron::SpikingNeuron
  weight::Float64
  delay::Int # number of periods to delay the pulse
end

function fire!(synapse::DelaySynapse)
  synapse.postsynapticNeuron.futurePulses[synapse.delay] += synapse.weight
end

Base.print(io::IO, s::DelaySynapse) = print(io, "DelaySynapse{post=$(s.postsynapticNeuron), wgt=$(s.weight), delay=$(s.delay)}")
Base.show(io::IO, s::DelaySynapse) = print(io, s)

# ---------------------------------------------------------------------

# we use the model [ uₜ = pulsesₜ + (1-λ) * uₜ₋₁ ] to update
type DiscreteLeakyIntegrateAndFireNeuron <: SpikingNeuron
  position::VecI  # 3x1 position in neuronal column
  excitatory::Bool
  futurePulses::CircularBuffer{Float64}  # future pulses by time offset {pulseₜ₊₁, pulseₜ₊₂, ...}
  u::Float64          # current state
  # uᵣₑₛₜ::Float64  # resting state # note: assume 0 for now
  ϑ::Float64      # threshold level
  decayRate::Float64        # exponential decay rate
  refractoryPeriodsRemaining::Int  # no activity allowed for this many periods after a spike
  refractoryPeriodsTotal::Int  # no activity allowed for this many periods after a spike
  fired::Bool  # did the neuron fire in the most recent period?
  synapses::Vector{DelaySynapse}
end

# TODO: make these parameters
const MAX_FUTURE = 5
const DEFAULT_THRESHOLD = 2.0
const DEFAULT_REFRACTORY_PERIOD = 2

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
                                      DelaySynapse[])
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
function fire!(neuron::SpikingNeuron)
  neuron.fired = neuron.u >= neuron.ϑ
  if neuron.fired
    foreach(neuron.synapses, fire!)
    neuron.u = 0.0
    neuron.refractoryPeriodsRemaining = neuron.refractoryPeriodsTotal
  end
end


function OnlineStats.state(neuron::SpikingNeuron)
  neuron.u / neuron.ϑ + 10.0 * float(neuron.fired)
end

