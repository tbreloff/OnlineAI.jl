
# one layer of a multilayer perceptron (neural net)
# ninᵢ := noutᵢ₋₁ + 1  (we add the bias term automatically, so there's one extra input)
# forward value is f(wx + b), where f is the activation function
# Σ := wx + b
type Layer{A <: Activation}
  nin::Int
  nout::Int
  activation::A
  p::Float64  # dropout retention probability

  # the state of the layer
  x::VecF  # nin x 1 -- input 
  w::MatF  # nout x nin -- weights connecting previous layer to this layer
  dw::MatF # nout x nin -- last changes in the weights (used for momentum)
  b::VecF  # nout x 1 -- bias terms
  db::VecF # nout x 1 -- last changes in bias terms (used for momentum)
  δ::VecF  # nout x 1 -- sensitivities (calculated during backward pass)
  Σ::VecF  # nout x 1 -- inner products (calculated during forward pass)
  r::VecF  # nin x 1 -- vector of dropout retention... 0 if we drop this incoming weight, 1 if we keep it
end

initialWeights(nin::Int, nout::Int) = (rand(nout, nin) - 0.5) * 2.0 * sqrt(6.0 / (nin + nout))

function Layer(nin::Integer, nout::Integer, activation::Activation, p::Float64 = 1.0)
  # w = hcat(initialWeights(nin, nout, zeros(nout))  # TODO: more generic initialization
  # nin += 1  # account for bias term
  # Layer(nin, nout, activation, zeros(nin), w, zeros(nout, nin), zeros(nout), zeros(nout))

  w = initialWeights(nin, nout)
  Layer(nin, nout, activation, p, zeros(nin), w, zeros(nout, nin), [zeros(nout) for i in 1:5]...)
end

Base.print(io::IO, l::Layer) = print(io, "Layer{$(l.nin)=>$(l.nout)}")


# takes input vector, and computes Σⱼ = wⱼ'x + bⱼ  and  Oⱼ = A(Σⱼ)
function forward(layer::Layer, x::AVecF, istraining::Bool)
  # layer.x = addOnes(x)
  layer.x = collect(x)
  # layer.r = istraining ? float(rand(layer.nin) .<= layer.p) : ones(layer.nin)  # apply dropout
  # layer.Σ = layer.w * (layer.x .* layer.r) + layer.b  # inner product
  if istraining
    layer.r = float(rand(layer.nin) .<= layer.p)
    layer.Σ = layer.w * (layer.x .* layer.r) + layer.b
  else
    layer.r = ones(layer.nin)
    layer.Σ = layer.p * (layer.w * layer.x) + layer.b
  end

  forward(layer.activation, layer.Σ)     # activate
end


# backward step for the final (output) layer
# TODO: this should be generalized to different loss functions
function updateSensitivities(layer::Layer, err::AVecF)
  layer.δ = -err .* backward(layer.activation, layer.Σ)
end

# this is the backward step for a hidden layer
# notes: we are figuring out the effect of each node's activation value on the next sensitivities
function updateSensitivities(layer::Layer, nextlayer::Layer)
  layer.δ = nextlayer.r .* (nextlayer.w' * nextlayer.δ) .* backward(layer.activation, layer.Σ)
end

function updateWeights(layer::Layer, solver::NNetSolver)
  # ΔW is a function which takes the gradients of the inputs, the weights and last weight change,
  # then computes the update to the weight matrix
  dw = ΔW(solver, layer.δ * (layer.x .* layer.r)', layer.w, layer.dw)
  layer.w += dw
  layer.dw = dw

  db = Δb(solver, layer.δ, layer.db)
  layer.b += db
  layer.db = db
end


