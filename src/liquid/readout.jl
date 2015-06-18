
# an implementation of Readout must implement the liquidState method which returns a VecF
abstract Readout


# simple readout of the current firings in the output neurons
type FireReadout <: Readout end
function liquidState(lsm, readout::FireReadout)
  Float64[float(neuron.fired) for neuron in lsm.liquid.outputNeurons]
end

# takes the state of the neuron, whatever that may be
type StateReadout <: Readout end
function liquidState(lsm, readout::StateReadout)
  Float64[float(state(neuron)) for neuron in lsm.liquid.outputNeurons]
end

