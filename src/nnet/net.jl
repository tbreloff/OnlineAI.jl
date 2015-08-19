
type NeuralNet <: NNetStat
  layers::Vector{Layer}  # note: this doesn't include input layer!!
  solver::NNetSolver
end


# simple constructor which creates all layers the same for given list of node counts.
# structure should include neuron counts for all layers, including input and output
function NeuralNet(structure::AVec{Int}; solver = NNetSolver(), activation::Activation = TanhActivation())
  @assert length(structure) > 1

  layers = Layer[]
  for i in 1:length(structure)-1
    nin, nout = structure[i:i+1]
    pDropout = getDropoutProb(solver, i==1)
    push!(layers, Layer(nin, nout, activation, pDropout))
  end

  NeuralNet(layers, solver)
end

function Base.show(io::IO, net::NeuralNet)
  println(io, "NeuralNet{solver=$(net.solver), layers:")
  for layer in net.layers
    println(io, "    ", layer)
  end
  println(io, "}")
end
Base.print(io::IO, net::NeuralNet) = show(io, net)


# produces a vector of yhat (estimated outputs) from the network
function forward(net::NeuralNet, x::AVecF, istraining::Bool = false)
  yhat = x
  for layer in net.layers
    yhat = forward(layer, yhat, istraining)
  end

  # update nextr
  for i in 1:length(net.layers)-1
    net.layers[i].nextr = net.layers[i+1].r
  end

  yhat
end


# given a vector of errors (y - yhat), update network weights
function backward(net::NeuralNet, errors::AVecF)

  # update δᵢ starting from the output layer
  updateSensitivities(net.layers[end], errors)
  for i in length(net.layers)-1:-1:1
    updateSensitivities(net.layers[i:i+1]...)
  end

  # now update the weights
  for layer in net.layers
    updateWeights(layer, net.solver)
  end
end


function totalerror(net::NeuralNet, x::AVecF, y::AVecF)
  yhat = forward(net, x)
  0.5 * sumabs2(y - yhat)
end


# online version... returns the feedforward estimate before updating
function OnlineStats.update!(net::NeuralNet, x::AVecF, y::AVecF)
  yhat = forward(net, x, true)
  errors = y - yhat
  backward(net, errors)
  yhat
end


# batch version
function OnlineStats.update!(net::NeuralNet, x::MatF, y::MatF)
  @assert ncols(x) == net.nin
  @assert ncols(y) == net.nout
  @assert nrows(x) == nrows(y)

  Float64[update!(net, row(x,i), row(y,i)) for i in 1:nrows(x)]
end

function StatsBase.predict(net::NeuralNet, x::AVecF)
  forward(net, x)
end

function StatsBase.predict(net::NeuralNet, x::AMatF)
  yhat = zeros(nrows(x), net.layers[end].nout)
  for i in 1:nrows(x)
    row!(yhat, i, predict(net, row(x, i)))
  end
  yhat
end



