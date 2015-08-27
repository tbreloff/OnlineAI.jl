
# one layer of a multilayer perceptron (neural net)
# ninᵢ := noutᵢ₋₁ + 1  (we add the bias term automatically, so there's one extra input)
# forward value is f(wx + b), where f is the activation function
# Σ := wx + b
# note: w is a parameter for the case of tied weights (it can be a TransposeView!)
type Layer{A <: Activation, MATF <: AbstractMatrix{Float64}, GSTATE <: GradientState}
  nin::Int
  nout::Int
  activation::A
  gradientState::GSTATE
  p::Float64  # dropout retention probability

  # the state of the layer
  x::VecF  # nin x 1 -- input 
  w::MATF  # nout x nin -- weights connecting previous layer to this layer
  # dw::MatF # nout x nin -- last changes in the weights (used for momentum)
  b::VecF  # nout x 1 -- bias terms
  # db::VecF # nout x 1 -- last changes in bias terms (used for momentum)
  δ::VecF  # nout x 1 -- sensitivities (calculated during backward pass)
  Σ::VecF  # nout x 1 -- inner products (calculated during forward pass)
  r::VecF  # nin x 1 -- vector of dropout retention... 0 if we drop this incoming weight, 1 if we keep it
  nextr::VecF  # nout x 1 -- retention of the nodes of this layer (as opposed to r which applies to the incoming weights)

  # Gw::MATF  # nout x nin -- sum of squares of the weight gradients, used for AdaGrad step
  # Gb::VecF  # nout x nin -- sum of squares of the bias gradients, used for AdaGrad step

end

initialWeights(nin::Int, nout::Int, activation::Activation) = (rand(nout, nin) - 0.5) * 2.0 * sqrt(6.0 / (nin + nout))
# initialWeights(nin::Int, nout::Int, activation::Activation) = randn(nout, nin) * 0.1

function Layer(nin::Integer, nout::Integer, activation::Activation, gradientModel::GradientModel, p::Float64 = 1.0)
  w = initialWeights(nin, nout, activation)
  gradientState = getGradientState(gradientModel, nin, nout)
  Layer(nin, nout, activation, gradientState, p, zeros(nin), w, [zeros(nout) for i in 1:3]..., ones(nin), ones(nout))
  # Layer(nin, nout, activation, p, zeros(nin), w, zeros(nout, nin), [zeros(nout) for i in 1:4]..., ones(nin), ones(nout), zeros(nout,nin), zeros(nout))
end

Base.print(io::IO, l::Layer) = print(io, "Layer{$(l.nin)=>$(l.nout) $(l.activation) p=$(l.p) ‖δ‖₁=$(sumabs(l.δ))}")


# takes input vector, and computes Σⱼ = wⱼ'x + bⱼ  and  Oⱼ = A(Σⱼ)
function forward(layer::Layer, x::AVecF, istraining::Bool)

  if istraining
    # train... randomly drop out incoming nodes
    # note: I said incoming, since this layers weights are the weights connecting the previous layer to this one
    #       So on dropout, we are actually dropping out thr previous layer's nodes...
    for i in 1:layer.nin
      ri = rand() <= layer.p ? 1.0 : 0.0
      layer.r[i] = ri
      layer.x[i] *= ri
    end
    # layer.r = float(rand(layer.nin) .<= layer.p)
    # layer.x = layer.r .* x
    layer.Σ = layer.w * layer.x + layer.b
  else
    # test... need to multiply weights by dropout prob p
    # layer.x = collect(x)
    # layer.r = ones(layer.nin)
    copy!(layer.x, x)
    fill!(layer.r, 1.0)
    layer.Σ = layer.p * (layer.w * layer.x) + layer.b
  end

  forward(layer.activation, layer.Σ)     # activate
end


# backward step for the final (output) layer
# note: costMult is the amount to multiply against f'(Σ)... L2 case should be: (yhat-y)
function updateSensitivities(layer::Layer, costMult::AVecF, multiplyDerivative::Bool)
  layer.δ = multiplyDerivative ? costMult .* backward(layer.activation, layer.Σ) : costMult
end

# this is the backward step for a hidden layer
# notes: we are figuring out the effect of each node's activation value on the next sensitivities
function updateSensitivities(layer::Layer, nextlayer::Layer)
  layer.δ = (nextlayer.w' * (nextlayer.nextr .* nextlayer.δ)) .* backward(layer.activation, layer.Σ)
end

# update weights/bias one column at a time... skipping over the dropped out nodes
# note we are tracking the sum of squares of gradients for use in the AdaGrad update,
# where we swap out the gradient `g` for the ratio `g / sqrt(G)`
# function updateWeights(layer::Layer, params::NetParams)
function updateWeights(layer::Layer, gradientModel::GradientModel)

  # note: i refers to the output, j refers to the input

  for i in 1:layer.nout

    if layer.nextr[i] > 0.0
      
      # if this node is retained, we can update incoming bias
      bGrad = layer.δ[i]  # δi is the gradient
      dbi = Δbi(gradientModel, layer.gradientState, bGrad, layer.b[i], i)
      # Gbi = layer.Gb[i] + δi^2

      # dbi = Gbi > 0.0 ? Δbi(params, δi / (params.useAdagrad ? sqrt(1.0+Gbi) : 1.0), layer.db[i]) : 0.0

      layer.b[i] += dbi
      # layer.db[i] = dbi
      # layer.Gb[i] = Gbi
      
      for j in 1:layer.nin
        
        # if this input node is retained, then we can also update the weight
        if layer.r[j] > 0.0
          
          wGrad = bGrad * layer.x[j]
          dwij = Δwij(gradientModel, layer.gradientState, wGrad, layer.w[i,j], i, j)
          # Gwij = layer.Gw[i,j] + wGrad^2
          
          # dwij = Gwij > 0.0 ? Δbij(params, wGrad / (params.useAdagrad ? sqrt(1.0+Gwij) : 1.0), layer.w[i,j], layer.dw[i,j]) : 0.0

          layer.w[i,j] += dwij
          # layer.dw[i,j] = dwij
          # layer.Gw[i,j] = Gwij
        end
      end
    end
  end
end


