
type NeuralNet <: NetStat
  layers::Vector{Layer}  # note: this doesn't include input layer!!
  params::NetParams
  solverParams::SolverParams
  inputTransformer::Transformer
  transformedInput::VecF   # so we can avoid allocations
  costmult::VecF

  # TODO: inner constructor which performs some sanity checking on activation/cost combinations:
  function NeuralNet(layers::Vector{Layer}, params::NetParams, solverParams::SolverParams, inputTransformer::Transformer = IdentityTransformer())

    # do some sanity checking on activation/costmodel combos
    if isa(params.costModel, CrossEntropyCostModel)
      @assert isa(layers[end].activation, layers[end].nout > 1 ? SoftmaxActivation : SigmoidActivation)
    end

    if isa(layers[end].activation, SoftmaxActivation)
      @assert isa(params.costModel, CrossEntropyCostModel)
    end

    new(layers, params, solverParams, inputTransformer, zeros(first(layers).nin), zeros(last(layers).nout))
  end
end


# simple constructor which creates all layers the same for given list of node counts.
# structure should include neuron counts for all layers, including input and output
function NeuralNet(structure::AVec{Int};
                   params = NetParams(),
                   solverParams = SolverParams(),
                   activation::Activation = TanhActivation(),
                   inputTransformer::Transformer = IdentityTransformer())
  @assert length(structure) > 1

  layers = Layer[]
  for i in 1:length(structure)-1
    nin, nout = structure[i:i+1]
    pDropout = getDropoutProb(params, i==1)
    push!(layers, Layer(nin, nout, activation, params.gradientModel, pDropout))
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
function forward!(net::NeuralNet, x::AVecF, istraining::Bool = false)
  
  # first transform the input
  transform!(net.inputTransformer, net.transformedInput, x)

  # now feed it forward
  yhat = net.transformedInput
  for layer in net.layers
    forward!(layer, yhat, istraining)
    yhat = layer.a
  end

  # update nextr
  for i in 1:length(net.layers)-1
    net.layers[i].nextr = net.layers[i+1].r
  end

  yhat
end


# given a vector of errors (y - yhat), update network weights
function backward(net::NeuralNet, multiplyDerivative::Bool)

  # update δᵢ starting from the output layer using the error multiplier
  updateSensitivities!(net.layers[end], net.costmult, multiplyDerivative)

  # now update the remaining sensitivities using bakckprop
  for i in length(net.layers)-1:-1:1
    updateSensitivities!(net.layers[i:i+1]...)
  end

  # now update the weights
  for layer in net.layers
    updateWeights!(layer, net.params.gradientModel)
  end

  # # update our η, μ, etc
  # update!(net.params)
end


# ------------------------------------------------------------------------


# online version... returns the feedforward estimate before updating
function OnlineStats.update!(net::NeuralNet, x::AVecF, y::AVecF)
  yhat = forward!(net, x, true)
  multiplyDerivative = costMultiplier!(net.params.costModel, net.costmult, y, yhat)
  backward!(net, multiplyDerivative)
  yhat
end


# batch version
function OnlineStats.update!(net::NeuralNet, x::MatF, y::MatF)
  @assert ncols(x) == net.nin
  @assert ncols(y) == net.nout
  @assert nrows(x) == nrows(y)

  Float64[update!(net, row(x,i), row(y,i)) for i in 1:nrows(x)]
end

# note: when yEqualsX is true, we are updating an autoencoder (or similar) and so we can
# use net.transformedInput instead of y
function OnlineStats.update!(net::NetStat, data::DataPoint, yEqualsX::Bool = false)
  update!(net, data.x, yEqualsX ? net.transformedInput : data.y)
  # update!(net, data.x, transformY ? transform(net.inputTransformer, data.y) : data.y)
end

# ------------------------------------------------------------------------

function cost(net::NeuralNet, x::AVecF, y::AVecF)
  yhat = forward!(net, x)
  cost(net.params.costModel, y, yhat)
end
totalCost(net::NetStat, data::DataPoint) = cost(net, data.x, data.y)
totalCost(net::NetStat, dataset::DataPoints) = sum([totalCost(net, data) for data in dataset])
totalCost(net::NetStat, sampler::DataSampler) = totalCost(net, DataPoints(sampler))

# ------------------------------------------------------------------------

function StatsBase.predict(net::NeuralNet, x::AVecF)
  forward!(net, x)
end

function StatsBase.predict(net::NeuralNet, x::AMatF)
  yhat = zeros(nrows(x), net.layers[end].nout)
  for i in 1:nrows(x)
    row!(yhat, i, predict(net, row(x, i)))
  end
  yhat
end



