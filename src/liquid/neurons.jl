
# Leaky Integrate and Fire (LIF)

type LIFSynapse <: Synapse
  postsynapticNeuron::SpikingNeuron
  weight::Float64
end

# transmit pulses to postsynaptic neurons
function fire!(synapse::LIFSynapse)
  neuron = synapse.postsynapticNeuron

  # make sure we won't send current negative with negative weights
  pulse = max(neuron.q, synapse.weight / τq)
  
  # increase pulse of the postsynaptic neuron
  neuron.q += pulse

  # add pulse to membrane potential (unless it just fired)
  if !neuron.fired
    neuron.u += pulse
  end
end

Base.print(io::IO, s::LIFSynapse) = print(io, "LIFSynapse{post=$(s.postsynapticNeuron), wgt=$(s.weight)}")
Base.show(io::IO, s::LIFSynapse) = print(io, s)


# ---------------------------------------------------------------------

# dirac pulse from presynaptic neuron (times weight) adds to total current of postsynaptic neuron
# total current decays back to 0 at rate τs (spike)
# (current * dt) is added to membrane potential (u) after decaying potential back to urest

# NOTE: to use OnlineStats ExponentialWeighting, instantiate with ExponentialWeighting(dt / τ)
#       where dt is the simulation time step, and τ is the "decay period"

const dt = 0.25         # ms - simulation time step
const τu = 30.0         # ms - membrane potential decay period
const τq = 4.0          # ms - pulse decay period
const urest = 0.0       # mV - resting membrane potential
const ufire = -0.4      # mV - membrane potential immediately after firing a spike
const baseThreshold = 2.0

type LIFNeuron <: SpikingNeuron
  position::VecI
  excitatory::Bool
  u::Float64      # membrane potential level (Vm)
  q::Float64      # pulse current level
  ϑ::Float64      # membrane potential threshold
  fired::Bool     # did it fire this period?
  synapses::Vector{LIFSynapse}
end

function LIFNeuron(pos::VecI, exitatory::Bool)
  LIFNeuron(pos, exitatory, urest, 0.0, baseThreshold, false, LIFSynapse[])
end

LIFNeuron() = LIFNeuron(Int[0,0,0], true)


Base.print(io::IO, n::LIFNeuron) = 
  print(io, "LIFNeuron{pos=$(n.position), exite=$(n.excitatory), u=$(n.u), q=$(n.q), fired=$(n.fired), nsyn=$(length(n.synapses))}")
Base.show(io::IO, n::LIFNeuron) = print(io, n)


# decay q and u, then add (q * dt) to u
function OnlineStats.update!(neuron::LIFNeuron)
  neuron.fired = false
  neuron.q = smooth(neuron.q, 0.0, dt / τq)
  neuron.u = smooth(neuron.u, urest, dt / τu) + neuron.q * dt
end

function fire!(neuron::LIFNeuron)
  if neuron.u >= neuron.ϑ
    neuron.fired = true
    neuron.u = ufire
    neuron.q = 0.0
    foreach(neuron.synapses, fire!)
  end
end

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

# # Discrete LIF (my design... not biologically plausible)

# # When a neuron nᵢ fires, it sends a pulse of wᵢⱼ to the dᵗʰ position of the circular buffer of future pulses for nⱼ.
# # In other words: at time t there is a spike in neuron nᵢ.  at time t+d we apply a pulse wᵢⱼ to neuron nⱼ


# # connects neurons together
# type DelaySynapse <: Synapse
#   postsynapticNeuron::SpikingNeuron
#   weight::Float64
#   delay::Int # number of periods to delay the pulse
# end

# function fire!(synapse::DelaySynapse)
#   synapse.postsynapticNeuron.futurePulses[synapse.delay] += synapse.weight
# end

# Base.print(io::IO, s::DelaySynapse) = print(io, "DelaySynapse{post=$(s.postsynapticNeuron), wgt=$(s.weight), delay=$(s.delay)}")
# Base.show(io::IO, s::DelaySynapse) = print(io, s)

# # ---------------------------------------------------------------------

# # we use the model [ uₜ = pulsesₜ + (1-λ) * uₜ₋₁ ] to update
# type DiscreteLeakyIntegrateAndFireNeuron <: SpikingNeuron
#   position::VecI  # 3x1 position in neuronal column
#   excitatory::Bool
#   futurePulses::CircularBuffer{Float64}  # future pulses by time offset {pulseₜ₊₁, pulseₜ₊₂, ...}
#   u::Float64          # current state
#   # uᵣₑₛₜ::Float64  # resting state # note: assume 0 for now
#   ϑ::Float64      # threshold level
#   decayRate::Float64        # exponential decay rate
#   refractoryPeriodsRemaining::Int  # no activity allowed for this many periods after a spike
#   refractoryPeriodsTotal::Int  # no activity allowed for this many periods after a spike
#   fired::Bool  # did the neuron fire in the most recent period?
#   synapses::Vector{DelaySynapse}
# end

# # TODO: make these parameters
# const MAX_FUTURE = 5
# const DEFAULT_THRESHOLD = 2.0
# const DEFAULT_REFRACTORY_PERIOD = 2

# function DiscreteLeakyIntegrateAndFireNeuron(position::VecI, excitatory::Bool, decayRate::Float64)
#   DiscreteLeakyIntegrateAndFireNeuron(position,
#                                       excitatory,
#                                       CircularBuffer(Float64, MAX_FUTURE, 0.0),
#                                       0.0,
#                                       DEFAULT_THRESHOLD,
#                                       decayRate,
#                                       0,
#                                       DEFAULT_REFRACTORY_PERIOD,
#                                       false,
#                                       DelaySynapse[])
# end

# Base.print(io::IO, n::DiscreteLeakyIntegrateAndFireNeuron) = 
#   print(io, "DLIFNeuron{pos=$(n.position), exite=$(n.excitatory), u=$(n.u), fired=$(n.fired), nsyn=$(length(n.synapses))}")
# Base.show(io::IO, n::DiscreteLeakyIntegrateAndFireNeuron) = print(io, n)

# # stepping through time involves 2 actions:
# # - incorporate pulses into u (spike --> reset and apply pulse to other neuron's futurePulses) and decay
# # - push! 0 onto futurePulses to step forward into time

# function OnlineStats.update!(neuron::DiscreteLeakyIntegrateAndFireNeuron)

#   # if we're in the refractory period, don't adjust u
#   if neuron.refractoryPeriodsRemaining > 0
#     neuron.refractoryPeriodsRemaining -= 1
#   else
#     neuron.u *= neuron.decayRate
#     neuron.u += neuron.futurePulses[1]
#   end

#   # don't let it go negative
#   neuron.u = max(0.0, neuron.u)

#   # step to next time period
#   push!(neuron.futurePulses, 0.0)
# end

# # fire/reset neurons that have crossed the threshold
# function fire!(neuron::SpikingNeuron)
#   neuron.fired = neuron.u >= neuron.ϑ
#   if neuron.fired
#     foreach(neuron.synapses, fire!)
#     neuron.u = 0.0
#     neuron.refractoryPeriodsRemaining = neuron.refractoryPeriodsTotal
#   end
# end


function OnlineStats.state(neuron::SpikingNeuron)
  neuron.u / neuron.ϑ + 10.0 * float(neuron.fired)
end

