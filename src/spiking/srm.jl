
# Spike Response Model neurons

type SRMSynapse <: Synapse
  postsynapticNeuron::SpikingNeuron
  weight::Float64
end

# transmit pulses to postsynaptic neurons
function fire!(synapse::SRMSynapse)
  psn = synapse.postsynapticNeuron
  
  # increase pulse of the postsynaptic neuron
  psn.q += synapse.weight * psn.qDecayRate

  # make sure we won't send current negative with negative weights
  psn.q = max(0.0, psn.q)
end

Base.print(io::IO, s::SRMSynapse) = print(io, "SRMSynapse{post=$(s.postsynapticNeuron), wgt=$(s.weight)}")
Base.show(io::IO, s::SRMSynapse) = print(io, s)


# ---------------------------------------------------------------------

# dirac pulse from presynaptic neuron (times weight) adds to total current of postsynaptic neuron
# total current decays back to 0 at rate τs (spike)
# (current * dt) is added to membrane potential (u) after decaying potential back to urest

# NOTE: to use OnlineStats ExponentialWeight, instantiate with ExponentialWeight(dt / τ)
#       where dt is the simulation time step, and τ is the "decay period"

# const dt = 0.25         # ms - simulation time step
# const τu = 25.0         # ms - membrane potential decay period
# const τq = 4.0          # ms - pulse decay period
# const urest = 0.0       # mV - resting membrane potential
# const ufire = -0.4      # mV - membrane potential immediately after firing a spike
# const baseThreshold = 2.0


"""
Spike Response Model (SRM) Neuron.  In my model, we update our membrane potential (u) by decaying u
towards urest with exponential weighting uDecayRate (equal to 1/τu).  At the same time we add to u:
    'integral over qₜ --> qₜ₊₁'
where q also decays towards 0.0 with exponential weighting qDecayRate (equal to 1/τq)
"""
type SRMNeuron <: SpikingNeuron
  position::VecI
  excitatory::Bool
  u::Float64      # membrane potential level (Vm)
  q::Float64      # pulse current level
  ϑ::Float64      # membrane potential threshold
  fired::Bool     # did it fire this period?
  synapses::Vector{SRMSynapse}
  
  # τu::Float64
  # τq::Float64
  uDecayRate::Float64
  qDecayRate::Float64
  urest::Float64
  ufire::Float64
end

function SRMNeuron(pos::VecI, exitatory::Bool, params)
  SRMNeuron(pos, exitatory, params.urest, 0.0, params.baseThreshold, false, SRMSynapse[], params.uDecayRate, params.qDecayRate, params.urest, params.ufire)
end

SRMNeuron(params) = SRMNeuron(Int[0,0,0], true, params)


Base.print(io::IO, n::SRMNeuron) = 
  print(io, "SRMNeuron{pos=$(n.position), exite=$(n.excitatory), u=$(n.u), q=$(n.q), fired=$(n.fired), nsyn=$(length(n.synapses))}")
Base.show(io::IO, n::SRMNeuron) = print(io, n)


# decay q and u, then add (q * dt) to u
function OnlineStats.fit!(neuron::SRMNeuron, dt::Float64)
  neuron.fired = false

  if neuron.qDecayRate > 0.0
    # decay the pulse, but save the first value so we can calculate the value of the integral of q over this time interval
    prevq = neuron.q
    neuron.q = OnlineStats.smooth(neuron.q, 0.0, dt * neuron.qDecayRate)

    # update neuron.u += du, where du is the integral of the pulse equation during the previous "dt" ms
    # note: use the difference of integrals of the pulse from t -> Inf, and t+1 -> Inf
    du = (prevq - neuron.q) / neuron.qDecayRate
  else
    # constant pulse over this time interval
    du = neuron.q * dt
  end

  neuron.u = OnlineStats.smooth(neuron.u, neuron.urest, dt * neuron.uDecayRate) + du
end

function fire!(neuron::SRMNeuron)
  if neuron.u >= neuron.ϑ
    neuron.fired = true
    neuron.u = neuron.ufire
    neuron.q = 0.0
    foreach(neuron.synapses, fire!)
  end
end


function OnlineStats.value(neuron::SpikingNeuron)
  neuron.u / neuron.ϑ + 10.0 * float(neuron.fired)
end

