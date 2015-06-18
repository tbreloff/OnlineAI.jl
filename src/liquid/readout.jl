
# an implementation of Readout must implement the liquidState method which returns a VecF
abstract Readout


# simple readout of the current firings in the output neurons
type FireReadout <: Readout end
function liquidState(lsm, readout::FireReadout)
  Float64[float(neuron.fired) for neuron in lsm.liquid.outputNeurons]
end

