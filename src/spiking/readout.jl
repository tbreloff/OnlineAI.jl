
# an implementation of Readout must implement the liquidState method which returns a VecF
abstract Readout

# default is to do nothing on update
OnlineStats.fit!{T <: SpikingNeuron}(readout::Readout, neurons::AbstractVector{T}) = nothing

# ----------------------------------------------------

# simple readout of the current firings in the output neurons
type FireReadout <: Readout end

function liquidState(lsm, readout::FireReadout)
  Float64[float(neuron.fired) for neuron in lsm.liquid.outputNeurons]
end

# ----------------------------------------------------

# takes the state of the neuron, whatever that may be
type StateReadout <: Readout end

function liquidState(lsm, readout::StateReadout)
  Float64[float(state(neuron)) for neuron in lsm.liquid.outputNeurons]
end

# ----------------------------------------------------

# keep a history of firings for each neuron
type FireWindowReadout <: Readout
  lookbackWindow::Int
  firings::Vector{CircularBuffer{Float64}}  # one CB per readout neuron
  FireWindowReadout(lookbackWindow::Integer) = new(lookbackWindow, CircularBuffer{Float64}[])
end

function OnlineStats.fit!{T <: SpikingNeuron}(readout::Readout, neurons::AbstractVector{T})
  
  # initialize on first pass
  if isempty(readout.firings)
    readout.firings = [CircularBuffer(Float64, readout.lookbackWindow) for i in 1:length(neurons)]
  end

  # add to window
  for (i,neuron) in enumerate(neurons)
    push!(readout.firings[i], Float64(neuron.fired))
  end
end

function liquidState(lsm, readout::FireWindowReadout)
  Float64[sum(cb) for cb in readout.firings]
end