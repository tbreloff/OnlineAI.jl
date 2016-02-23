
# one layer of a multilayer perceptron (neural net)
# ninᵢ := noutᵢ₋₁ + 1  (we add the bias term automatically, so there's one extra input)
# forward value is f(wx + b), where f is the activation function
# Σ := wx + b
# note: w is a parameter for the case of tied weights (it can be a TransposeView!)
type Layer{A <: Activation, MATF <: AbstractMatrix{Float64}, GSTATE <: GradientState} <: NeuralNetLayer
  nin::Int
  nout::Int
  activation::A
  p::Float64  # dropout retention probability

  # the state of the layer
  
  dwState::GSTATE
  dbState::GSTATE  

  x::VecF  # nin x 1 -- input 

  w::MATF  # nout x nin -- weights connecting previous layer to this layer
  b::VecF  # nout x 1 -- bias terms
  δ::VecF  # nout x 1 -- sensitivities (calculated during backward pass)
  Σ::VecF  # nout x 1 -- inner products (calculated during forward pass)
  a::VecF  # nout x 1 -- f(Σ) (calculated during forward pass)

  r::VecF  # nin x 1 -- vector of dropout retention... 0 if we drop this incoming weight, 1 if we keep it
  nextr::VecF  # nout x 1 -- retention of the nodes of this layer (as opposed to r which applies to the incoming weights)
end


function Layer(nin::Integer, nout::Integer, activation::Activation,
               gradientModel::GradientModel, p::Float64 = 1.0;
               wgt = nothing, weightInit::Function = _initialWeights)
  w = weightInit(nin, nout, activation)
  Layer(nin, nout, activation, p,
            gradient_state(gradientModel, nout, nin),
            gradient_state(gradientModel, nout, 1),
            zeros(nin),
            w,
            [zeros(nout) for i in 1:4]...,
            ones(nin),
            ones(nout))
end

function Base.print{A,M,G}(io::IO, l::Layer{A,M,G})
  print(io, "Layer{$(l.nin)=>$(l.nout) $(l.activation) p=$(l.p) ‖δ‖₁=$(sumabs(l.δ)) $(M<:TransposeView ? "T" : "")}")
end


# gemv! :: Σ += p * w * x  (note: 'T' would imply p * w' * x)
@generated function dosigmamult!(layer::Layer, α::Float64)
  if layer.parameters[2] <: TransposeView
    return quote
      copy!(layer.Σ, layer.b)
      BLAS.gemv!('T', α, layer.w.mat, layer.x, 1.0, layer.Σ)
    end
  else
    return quote
      copy!(layer.Σ, layer.b)
      BLAS.gemv!('N', α, layer.w, layer.x, 1.0, layer.Σ)
    end
  end
end

# takes input vector, and computes Σⱼ = wⱼ'x + bⱼ  and  Oⱼ = A(Σⱼ)
function forward!(layer::Layer, x::AVecF, istraining::Bool)

  if istraining
    # train... randomly drop out incoming nodes
    # note: I said incoming, since this layers weights are the weights connecting the previous layer to this one
    #       So on dropout, we are actually dropping out thr previous layer's nodes...
    for i in 1:layer.nin
      ri = rand() <= layer.p ? 1.0 : 0.0
      layer.r[i] = ri
      layer.x[i] = ri * x[i]
    end

    dosigmamult!(layer, 1.0)
  else
    # test... need to multiply weights by dropout prob p
    fill!(layer.r, 1.0)
    copy!(layer.x, x)

    dosigmamult!(layer, layer.p)
  end

  forward!(layer.activation, layer.a, layer.Σ)     # activate
end


# backward step for the final (output) layer
# note: costmult is the amount to multiply against f'(Σ)... L2 case should be: (yhat-y)
function updateSensitivities!(layer::Layer, costmult::AVecF, multiplyDerivative::Bool)
  copy!(layer.δ, costmult)
  if multiplyDerivative
    for i in 1:layer.nout
      layer.δ[i] *= backward(layer.activation, layer.Σ[i])
    end
  end
end

# this is the backward step for a hidden layer
# notes: we are figuring out the effect of each node's activation value on the next sensitivities
function updateSensitivities!(layer::Layer, nextlayer::Layer)
  for i in 1:layer.nout
    δi = 0.0
    for j in 1:nextlayer.nout
      δi += nextlayer.w[j,i] * nextlayer.nextr[j] * nextlayer.δ[j]
    end
    layer.δ[i] = δi * backward(layer.activation, layer.Σ[i])
  end
end

# update weights/bias one column at a time... skipping over the dropped out nodes
function updateWeights!(layer::Layer, gradientModel::GradientModel)

  # note: i refers to the output, j refers to the input

  for i in 1:layer.nout

    if layer.nextr[i] > 0.0
      
      # if this node is retained, we can update incoming bias
      bGrad = layer.δ[i]  # δi is the gradient
      dbi = Δij(gradientModel, layer.dbState, bGrad, layer.b[i], i, 1)

      layer.b[i] += dbi
      
      for j in 1:layer.nin
        
        # if this input node is retained, then we can also update the weight
        if layer.r[j] > 0.0
          
          wGrad = bGrad * layer.x[j]
          dwij = Δij(gradientModel, layer.dwState, wGrad, layer.w[i,j], i, j)

          layer.w[i,j] += dwij
        end
      end
    end
  end
end


