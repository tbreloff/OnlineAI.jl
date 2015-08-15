# solver should contain all algorithm-specific parameters and methods.
# at a minimum, we need to be able to compute the weight updates for a layer

type NNetSolver
  η::Float64 # learning rate
  μ::Float64 # momentum
  λ::Float64 # L2 penalty term
end

NNetSolver(; η=1e-2, μ=0.0, λ=0.0001) = NNetSolver(η, μ, λ)

# calc update to weight matrix.  TODO: generalize penalty
function ΔW(solver::NNetSolver, gradients::AVecF, w::AMatF, dw::AMatF)
  -solver.η * (gradients + solver.λ * w) + solver.μ * dw
end

# -------------------------------------

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
    push!(layers, Layer(nin, nout, activation))
  end

  NeuralNet(layers, NNetSolver(η, μ))
end

function Base.show(io::IO, net::NeuralNet)
  println(io, "NeuralNet{η=$(net.η), μ=$(net.μ), layers:")
  for layer in net.layers
    println(io, "    ", layer)
  end
  println(io, "}")
end


# produces a vector of yhat (estimated outputs) from the network
function forward(net::NeuralNet, x::AVecF)
  yhat = x
  for layer in net.layers
    yhat = forward(layer, yhat)
  end
  yhat
end


# given a vector of errors (y - yhat), update network weights
function backward(net::NeuralNet, errors::AVecF)

  # update δ (sensitivities)
  finalδ!(net.layers[end], errors)
  for i in length(net.layers)-1:-1:1
    hiddenδ!(net.layers[i], net.layers[i+1])
  end

  # update weights
  for layer in net.layers
    update!(layer, net.η, net.μ)
  end

end


function totalerror(net::NeuralNet, x::AVecF, y::AVecF)
  yhat = forward(net, x)
  0.5 * sumabs2(y - yhat)
end


# online version
function OnlineStats.update!(net::NeuralNet, x::AVecF, y::AVecF)
  yhat = forward(net, x)
  errors = y - yhat
  backward(net, errors)
  yhat
end


# batch version
function OnlineStats.update!(net::NeuralNet, x::MatF, y::MatF)
  @assert size(x,2) == net.nin
  @assert size(y,2) == net.nout
  @assert size(x,1) == size(y,1)

  yhat = AVecF[]
  for i in 1:size(x,1)
    output = update!(net, row(x,i), row(y,i))
    push!(yhat, output)
  end
  yhat
end

