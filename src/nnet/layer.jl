
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
  used::Vector{Bool}  # nout x 1 -- is each node retained?  this applies to this layer's nodes, where r applies to incoming weights
end

initialWeights(nin::Int, nout::Int, activation::Activation) = (rand(nout, nin) - 0.5) * 2.0 * sqrt(6.0 / (nin + nout))
initialWeights(nin::Int, nout::Int, activation::Union(ReLUActivation,LReLUActivation)) = zeros(nout, nin)

function Layer(nin::Integer, nout::Integer, activation::Activation, p::Float64 = 1.0)
  # w = hcat(initialWeights(nin, nout, zeros(nout))  # TODO: more generic initialization
  # nin += 1  # account for bias term
  # Layer(nin, nout, activation, zeros(nin), w, zeros(nout, nin), zeros(nout), zeros(nout))

  w = initialWeights(nin, nout, activation)
  Layer(nin, nout, activation, p, zeros(nin), w, zeros(nout, nin), [zeros(nout) for i in 1:5]..., fill(true, nout))
end

Base.print(io::IO, l::Layer) = print(io, "Layer{$(l.nin)=>$(l.nout) $(l.activation) p=$(l.p) r=$(l.r)}")


# takes input vector, and computes Σⱼ = wⱼ'x + bⱼ  and  Oⱼ = A(Σⱼ)
function forward(layer::Layer, x::AVecF, istraining::Bool)
  # layer.x = addOnes(x)
  # layer.x = collect(x)
  # layer.r = istraining ? float(rand(layer.nin) .<= layer.p) : ones(layer.nin)  # apply dropout
  # layer.Σ = layer.w * (layer.x .* layer.r) + layer.b  # inner product

  if istraining
    # train... randomly drop out incoming nodes
    # note: I said incoming, since this layers weights are the weights connecting the previous layer to this one
    #       So on dropout, we are actually dropping out thr previous layer's nodes...
    layer.r = float(rand(layer.nin) .<= layer.p)
    layer.x = layer.r .* x
    layer.Σ = layer.w * layer.x + layer.b
  else
    # test... need to multiply weights by dropout prob p
    layer.x = collect(x)
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
  layer.used = nextlayer.r .> 0.0
  layer.δ = nextlayer.r .* (nextlayer.w' * nextlayer.δ) .* backward(layer.activation, layer.Σ)
  # layer.δ = (nextlayer.w' * nextlayer.δ) .* backward(layer.activation, layer.Σ)
end

# TODO: update weights/bias one column at a time... skipping over the dropped out nodes
function updateWeights(layer::Layer, solver::NNetSolver)
  dw = ΔW(solver, layer.δ * layer.x', layer.w, layer.dw)
  db = Δb(solver, layer.δ, layer.db)
  for iIn in 1:layer.nin
    for iNode in 1:layer.nout
      if layer.used[iNode]
        if layer.r[iIn] > 0.0
          layer.w[iNode,iIn] += dw[iNode,iIn]
          layer.dw[iNode,iIn] = dw[iNode,iIn]
        end
        layer.b[iNode] += db[iNode]
        layer.db[iNode] = db[iNode]
      end
    end
  end
end

# function updateWeights(layer::Layer, solver::NNetSolver)
#   # ΔW is a function which takes the gradients of the inputs, the weights and last weight change,
#   # then computes the update to the weight matrix
#   # dw = ΔW(solver, layer.δ * (layer.x .* layer.r)', layer.w, layer.dw)
#   dw = ΔW(solver, layer.δ * layer.x', layer.w, layer.dw)
#   layer.w += dw
#   layer.dw = dw

#   db = Δb(solver, layer.δ, layer.db)
#   layer.b += db
#   layer.db = db
# end


