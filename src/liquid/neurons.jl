
# Leaky Integrate and Fire (LIF)

type LIFSynapse <: Synapse
  postsynapticNeuron::SpikingNeuron
  weight::Float64
end

# transmit pulses to postsynaptic neurons
function fire!(synapse::LIFSynapse)
  neuron = synapse.postsynapticNeuron
  
  # increase pulse of the postsynaptic neuron
  neuron.q += synapse.weight / neuron.τq

  # make sure we won't send current negative with negative weights
  neuron.q = max(0.0, neuron.q)

  # # add pulse to postsynaptic neuron (unless it just fired)
  # if !neuron.fired
  #   neuron.q += pulse
  #   neuron.u += pulse
  # end
end

Base.print(io::IO, s::LIFSynapse) = print(io, "LIFSynapse{post=$(s.postsynapticNeuron), wgt=$(s.weight)}")
Base.show(io::IO, s::LIFSynapse) = print(io, s)


# ---------------------------------------------------------------------

# dirac pulse from presynaptic neuron (times weight) adds to total current of postsynaptic neuron
# total current decays back to 0 at rate τs (spike)
# (current * dt) is added to membrane potential (u) after decaying potential back to urest

# NOTE: to use OnlineStats ExponentialWeighting, instantiate with ExponentialWeighting(dt / τ)
#       where dt is the simulation time step, and τ is the "decay period"

# const dt = 0.25         # ms - simulation time step
# const τu = 25.0         # ms - membrane potential decay period
# const τq = 4.0          # ms - pulse decay period
# const urest = 0.0       # mV - resting membrane potential
# const ufire = -0.4      # mV - membrane potential immediately after firing a spike
# const baseThreshold = 2.0

type LIFNeuron <: SpikingNeuron
  position::VecI
  excitatory::Bool
  u::Float64      # membrane potential level (Vm)
  q::Float64      # pulse current level
  ϑ::Float64      # membrane potential threshold
  fired::Bool     # did it fire this period?
  synapses::Vector{LIFSynapse}
  
  τu::Float64
  τq::Float64
  urest::Float64
  ufire::Float64
end

function LIFNeuron(pos::VecI, exitatory::Bool, params)
  LIFNeuron(pos, exitatory, params.urest, 0.0, params.baseThreshold, false, LIFSynapse[], params.τu, params.τq, params.urest, params.ufire)
end

LIFNeuron(params) = LIFNeuron(Int[0,0,0], true, params)


Base.print(io::IO, n::LIFNeuron) = 
  print(io, "LIFNeuron{pos=$(n.position), exite=$(n.excitatory), u=$(n.u), q=$(n.q), fired=$(n.fired), nsyn=$(length(n.synapses))}")
Base.show(io::IO, n::LIFNeuron) = print(io, n)


# integralToInf(q::Float64) = q * τq


# decay q and u, then add (q * dt) to u
function OnlineStats.update!(neuron::LIFNeuron, dt::Float64)
  neuron.fired = false


  # update the pulse
  prevq = neuron.q
  neuron.q = OnlineStats.smooth(neuron.q, 0.0, dt / neuron.τq)
  # dq = neuron.q * (1.0 - dt / neuron.τq)

  # TODO: update neuron.u += du, where du is the integral of the pulse equation during the previous "dt" ms
  # note: use the difference of integrals of the pulse from t -> Inf, and t+1 -> Inf
  # du = integralToInf(prevq) - integralToInf(neuron.q)
  # du = neuron.q * dt
  du = neuron.τq * (prevq - neuron.q)
  neuron.u = OnlineStats.smooth(neuron.u, neuron.urest, dt / neuron.τu) + du
end

function fire!(neuron::LIFNeuron)
  if neuron.u >= neuron.ϑ
    neuron.fired = true
    neuron.u = neuron.ufire
    neuron.q = 0.0
    foreach(neuron.synapses, fire!)
  end
end


function OnlineStats.state(neuron::SpikingNeuron)
  neuron.u / neuron.ϑ + 10.0 * float(neuron.fired)
end

