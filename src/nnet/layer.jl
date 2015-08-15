
# one layer of a multilayer perceptron (neural net)
# ninᵢ := noutᵢ₋₁ + 1  (we add the bias term automatically, so there's one extra input)
# forward value is f(wx + b), where f is the activation function
# Σ := wx + b
type Layer{A <: Activation}
  nin::Int
  nout::Int
  activation::A

  # the state of the layer
  x::VecF  # nin x 1 -- input 
  w::MatF  # nout x nin -- weights connecting previous layer to this layer
  dw::MatF # nout x nin -- last changes in the weights (used for momentum)
  # b::VecF  # nout x 1 -- bias terms
  # db::VecF # nout x 1 -- last changes in bias terms (used for momentum)
  δ::VecF  # nout x 1 -- sensitivities (calculated during backward pass)
  Σ::VecF  # nout x 1 -- inner products (calculated during forward pass)
end

function Layer(nin::Integer, nout::Integer, activation::Activation)
  w = hcat((rand(nout, nin) - 0.5) * 2.0 * sqrt(6.0 / (nin + nout)), zeros(nout))  # TODO: more generic initialization
  nin += 1  # account for bias term
  Layer(nin, nout, activation, zeros(nin), w, zeros(nout, nin), zeros(nout), zeros(nout))
end

Base.print(io::IO, l::Layer) = print(io, "Layer{$(l.nin)=>$(l.nout)}")


# initialWeights(nin::Int, nout::Int) = vcat((rand(nin) - 0.5) * 2.0 * sqrt(6.0 / (nin + nout)), 0.0)


# function buildLayer(nin::Int, nout::Int, activation::Activation = SigmoidActivation())
#   nodes = [Perceptron(initialWeights(nin, nout), activation) for j in 1:nout]
#   Layer(nin, nout, nodes)
# end



# takes input vector, and computes Σⱼ = wⱼ'x + bⱼ  and  Oⱼ = A(Σⱼ)
function forward(layer::Layer, x::AVecF)
  layer.x = vcat(collect(x), 1.0)
  layer.Σ = layer.w * layer.x  # inner product
  forward(layer.activation, layer.Σ)     # activate
end


# function hiddenδ!(node::Perceptron, weightedδ::Float64)
#   node.δ = node.dA(node.Σ) * weightedδ
# end

# function finalδ!(node::Perceptron, err::Float64)
#   node.δ = -node.dA(node.Σ) * err  # this probably needs to change for different loss functions?
# end

# function OnlineStats.update!(node::Perceptron, η::Float64, μ::Float64)
#   dw = -η * node.δ * node.x + μ * node.dw
#   node.w += dw
#   node.dw = dw
# end

# backward step for the final (output) layer
# TODO: this should be generalized to different loss functions
function updateSensitivities(layer::Layer, err::AVecF)
  layer.δ = -err .* backward(layer.activation, layer.Σ)
end

# this is the backward step for a hidden layer
# notes: we are figuring out the effect of each node's activation value on the next sensitivities
function updateSensitivities(layer::Layer, nextlayer::Layer)
  # map(x->println(size(x)), Any[nextlayer.w, nextlayer.δ, layer.Σ])
  layer.δ = (nextlayer.w' * nextlayer.δ)[1:end-1] .* backward(layer.activation, layer.Σ)
end

function updateWeights(layer::Layer, solver::NNetSolver)
  # ΔW is a function which takes the gradients of the inputs, the weights and last weight change,
  # then computes the update to the weight matrix
  dw = ΔW(solver, layer.δ * layer.x', layer.w, layer.dw)
  layer.w += dw
  layer.dw = dw
end


# δ(layer::Layer, j::Int) = layer.nodes[j].δ
# δ(layer::Layer) = Float64[δ(layer, j) for j in 1:layer.nout]

# function hiddenδ!(layer::Layer, nextlayer::Layer)
#   for j in 1:layer.nout
#     weightedNextδ = dot(δ(nextlayer), Float64[node.w[j] for node in nextlayer.nodes])
#     hiddenδ!(layer.nodes[j], weightedNextδ)
#   end
# end

# function finalδ!(layer::Layer, errors::VecF)
#   for j in 1:layer.nout
#     finalδ!(layer.nodes[j], errors[j])
#   end
# end

# function OnlineStats.update!(layer::Layer, η::Float64, μ::Float64)
#   for node in layer.nodes
#     update!(node, η, μ)
#   end
# end