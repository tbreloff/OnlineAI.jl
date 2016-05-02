
const LAYER = Layer
# const LAYER = NormalizedLayer

type NeuralNet <: NetStat
  layers::Vector{LAYER}  # note: this doesn't include input layer!!
  params::NetParams
  solverParams::SolverParams
  inputTransformer::Transformer
  transformedInput::VecF   # so we can avoid allocations
  # costmult::VecF

  # TODO: inner constructor which performs some sanity checking on activation/cost combinations:
  function NeuralNet(layers::Vector{LAYER},
                      params::NetParams,
                      solverParams::SolverParams,
                      inputTransformer::Transformer = IdentityTransformer())

    # do some sanity checking on activation/costmodel combos
    if isa(params.mloss, CrossentropyLoss)
      @assert isa(layers[end].activation, layers[end].nout > 1 ? SoftmaxMapping : SigmoidMapping)
    end

    if isa(layers[end].activation, SoftmaxMapping)
      @assert isa(params.mloss, CrossentropyLoss)
    end

    new(layers, params, solverParams, inputTransformer, zeros(first(layers).nin)) #, zeros(last(layers).nout))
  end
end


# # simple constructor which creates all layers the same for given list of node counts.
# # structure should include neuron counts for all layers, including input and output
# function NeuralNet(structure::AVec{Int};
#                     params = NetParams(),
#                     solverParams = SolverParams(),
#                     activation::Mapping = TanhMapping(),
#                     inputTransformer::Transformer = IdentityTransformer())
#   @assert length(structure) > 1

#   layers = LAYER[]
#   for i in 1:length(structure)-1
#     nin, nout = structure[i:i+1]
#     pDropout = getDropoutProb(params, i==1)
#     push!(layers, layerType(nin, nout, activation, params.updater, pDropout))
#   end

#   NeuralNet(layers, params, solverParams, inputTransformer)
# end

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

# produces a vector of output (estimated outputs) from the network
function forward!(net::NeuralNet, input::AVecF, istraining::Bool = false)

  # first transform the input
  transform!(net.inputTransformer, net.transformedInput, input)

  # now feed it forward
  output = net.transformedInput
  for layer in net.layers
    forward!(layer, output, istraining)
    output = layer.a
  end

  # update nextr
  for i in 1:length(net.layers)-1
    net.layers[i].nextr = net.layers[i+1].r
  end

  output
end


# given a vector of errors (target - output), update network weights
function backward!(net::NeuralNet, output, target)

  # update δᵢ starting from the output layer
  layer = net.layers[end]
  δ = sensitivity!(layer.δ, layer.activation, net.params.mloss, layer.x, output, target)

  # now update the remaining sensitivities using bakckprop
  for i in length(net.layers)-1:-1:1
    updateSensitivities!(net.layers[i:i+1]...)
  end

  # now update the weights
  for layer in net.layers
    updateWeights!(layer, net.params.updater)

    # println()
    # @show layer
  end

  # # update our η, μ, etc
  # fit!(net.params)
end


# ------------------------------------------------------------------------

# ϕₒ = sensitivity!(outputnode.state.ϕ,
#                   outputnode.mapping,
#                   net.mloss,
#                   input,
#                   output,
#                   target)

# online version... returns the feedforward estimate before updating
function OnlineStats.fit!(net::NeuralNet, input::AVecF, target::AVecF)
  output = forward!(net, input, true)
  backward!(net, output, target)
  output
end


# batch version
function OnlineStats.fit!(net::NeuralNet, input::MatF, target::MatF)
  @assert ncols(input) == net.nin
  @assert ncols(target) == net.nout
  @assert nrows(input) == nrows(target)

  Float64[fit!(net, row(input,i), row(target,i)) for i in 1:nrows(input)]
end

# note: when yEqualsX is true, we are updating an autoencoder (or similar) and so we can
# use net.transformedInput instead of target
function OnlineStats.fit!(net::NetStat, data::DataPoint, yEqualsX::Bool = false)
  fit!(net, data.x, yEqualsX ? net.transformedInput : data.y)
  # fit!(net, data.x, transformY ? transform(net.inputTransformer, data.y) : data.y)
end

# ------------------------------------------------------------------------

function cost(net::NeuralNet, input::AVecF, target::AVecF)
  output = forward!(net, input)
  sum(value(net.params.mloss, target, output))
end
totalCost(net::NetStat, data::DataPoint) = cost(net, data.x, data.y)
totalCost(net::NetStat, dataset::DataPoints) = sum([totalCost(net, data) for data in dataset])
totalCost(net::NetStat, sampler::DataSampler) = totalCost(net, DataPoints(sampler))

# ------------------------------------------------------------------------

function StatsBase.predict(net::NeuralNet, input::AVecF)
  forward!(net, input)
end

function StatsBase.predict(net::NeuralNet, input::AMatF)
  output = zeros(nrows(input), net.layers[end].nout)
  for i in 1:nrows(input)
    row!(output, i, predict(net, row(input, i)))
  end
  output
end

# ------------------------------------------------------------------------


# note: we scale standard random normals by (1/sqrt(nin)) so that the distribution of initial (Σ = wx + b)
#       is also approximately standard normal
_initialWeights(nin::Int, nout::Int, activation::Mapping = IdentityMapping()) = 0.5randn(nout, nin) / sqrt(nin)


# initialWeights(nin::Int, nout::Int, activation::Mapping) = (rand(nout, nin) - 0.5) * 2.0 * sqrt(6.0 / (nin + nout))

# note: we scale standard random normals by (1/sqrt(nin)) so that the distribution of initial (Σ = wx + b)
#       is also approximately standard normal
# initialWeights(nin::Int, nout::Int, activation::Mapping) = randn(nout, nin) / sqrt(nin)

# ------------------------------------------------------------------------
