
type NeuralNet <: NetStat
  layers::Vector{Layer}  # note: this doesn't include input layer!!
  params::NetParams
  solverParams::SolverParams
  inputTransformer::Function

  # TODO: inner constructor which performs some sanity checking on activation/cost combinations:
  # i.e. only allow cross entropy error with sigmoid activation
  function NeuralNet(layers::Vector{Layer}, params::NetParams, solverParams::SolverParams, inputTransformer::Function = nop)

    # do some sanity checking on activation/costmodel combos
    if isa(params.costModel, CrossEntropyCostModel)
      @assert isa(layers[end].activation, layers[end].nout > 1 ? SoftmaxActivation : SigmoidActivation)
    end

    new(layers, params, solverParams, inputTransformer)
  end
end


# simple constructor which creates all layers the same for given list of node counts.
# structure should include neuron counts for all layers, including input and output
function NeuralNet(structure::AVec{Int};
                   params = NetParams(),
                   solverParams = SolverParams(),
                   activation::Activation = TanhActivation(),
                   inputTransformer::Function = nop)
  @assert length(structure) > 1

  layers = Layer[]
  for i in 1:length(structure)-1
    nin, nout = structure[i:i+1]
    pDropout = getDropoutProb(params, i==1)
    push!(layers, Layer(nin, nout, activation, pDropout))
  end

  NeuralNet(layers, params, solverParams, inputTransformer)
end

function Base.show(io::IO, net::NeuralNet)
  println(io, "NeuralNet{")
  println(io, "  params: $(net.params)")
  println(io, "  solverParams: $(net.solverParams)")
  println(io, "  layers:")
  for layer in net.layers
    println(io, "    ", layer)
  end
  println(io, "}")
end
Base.print(io::IO, net::NeuralNet) = show(io, net)

# ------------------------------------------------------------------------

# produces a vector of yhat (estimated outputs) from the network
function forward(net::NeuralNet, x::AVecF, istraining::Bool = false)
  yhat = net.inputTransformer(x)
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
function backward(net::NeuralNet, errmult::AVecF, multiplyDerivative::Bool)

  # update δᵢ starting from the output layer using the error multiplier
  updateSensitivities(net.layers[end], errmult, multiplyDerivative)

  # now update the remaining sensitivities using bakckprop
  for i in length(net.layers)-1:-1:1
    updateSensitivities(net.layers[i:i+1]...)
  end

  # now update the weights
  for layer in net.layers
    updateWeights(layer, net.params)
  end

  # update our η, μ, etc
  update!(net.params)
end


# ------------------------------------------------------------------------


# online version... returns the feedforward estimate before updating
function OnlineStats.update!(net::NeuralNet, x::AVecF, y::AVecF)
  yhat = forward(net, x, true)
  errmult, multiplyDerivative = costMultiplier(net.params.costModel, y, yhat)
  backward(net, errmult, multiplyDerivative)
  yhat
end


# batch version
function OnlineStats.update!(net::NeuralNet, x::MatF, y::MatF)
  @assert ncols(x) == net.nin
  @assert ncols(y) == net.nout
  @assert nrows(x) == nrows(y)

  Float64[update!(net, row(x,i), row(y,i)) for i in 1:nrows(x)]
end

OnlineStats.update!(net::NetStat, data::DataPoint) = update!(net, data.x, data.y)

# ------------------------------------------------------------------------

function cost(net::NeuralNet, x::AVecF, y::AVecF)
  yhat = forward(net, x)
  cost(net.params.costModel, y, yhat)
end
totalCost(net::NetStat, data::DataPoint) = cost(net, data.x, data.y)
totalCost(net::NetStat, dataset::DataPoints) = sum([totalCost(net, data) for data in dataset])
totalCost(net::NetStat, sampler::DataSampler) = totalCost(net, DataPoints(sampler))

# ------------------------------------------------------------------------

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



