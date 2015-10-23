
# one layer of a multilayer perceptron (neural net)
# ninᵢ := noutᵢ₋₁ + 1  (we add the bias term automatically, so there's one extra input)
# forward value is f(wx + b), where f is the activation function
# Σ := wx + b
# note: w is a parameter for the case of tied weights (it can be a TransposeView!)
type Layer{A <: Activation, MATF <: AbstractMatrix{Float64}, GSTATE <: GradientState} <: NeuralNetLayer
  nin::Int
  nout::Int
  activation::A
  # gradientState::GSTATE
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

# initialWeights(nin::Int, nout::Int, activation::Activation) = (rand(nout, nin) - 0.5) * 2.0 * sqrt(6.0 / (nin + nout))

# note: we scale standard random normals by (1/sqrt(nin)) so that the distribution of initial (Σ = wx + b)
#       is also approximately standard normal
# initialWeights(nin::Int, nout::Int, activation::Activation) = randn(nout, nin) / sqrt(nin)

function Layer(nin::Integer, nout::Integer, activation::Activation, gradientModel::GradientModel, p::Float64 = 1.0; wgt=nothing)
  w = initialWeights(nin, nout, activation)
  # gradientState = getGradientState(gradientModel, nin, nout)
  # println(w)
  Layer(nin, nout, activation, p,
            getGradientState(gradientModel, nout, nin),
            getGradientState(gradientModel, nout, 1),
            zeros(nin),
            w,
            [zeros(nout) for i in 1:4]...,
            ones(nin),
            ones(nout))
  # Layer(nin, nout, activation, p, zeros(nin), w, zeros(nout, nin), [zeros(nout) for i in 1:4]..., ones(nin), ones(nout), zeros(nout,nin), zeros(nout))
end

Base.print{A,M,G}(io::IO, l::Layer{A,M,G}) = print(io, "Layer{$(l.nin)=>$(l.nout) $(l.activation) p=$(l.p) ‖δ‖₁=$(sumabs(l.δ)) $(M<:TransposeView ? "T" : "")}")


# # gemv! :: Σ += p * w * x  (note: 'T' would imply p * w' * x)
# function dosigmamult!{A,G}(layer::Layer{A,TransposeView{Float64},G}, α::Float64)
#   copy!(layer.Σ, layer.b)
#   BLAS.gemv!('T', α, layer.w.mat, layer.x, 1.0, layer.Σ)
# end
# function dosigmamult!(layer::Layer, α::Float64)
#   copy!(layer.Σ, layer.b)
#   BLAS.gemv!('N', α, layer.w, layer.x, 1.0, layer.Σ)
# end


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
    # layer.r = float(rand(layer.nin) .<= layer.p)
    # layer.x = layer.r .* x

    dosigmamult!(layer, 1.0)
    # copy!(layer.Σ, layer.b)
    # BLAS.gemv!('N', 1.0, layer.w, layer.x, 1.0, layer.Σ)    
    # layer.Σ = layer.w * layer.x + layer.b
  else
    # test... need to multiply weights by dropout prob p
    # layer.x = collect(x)
    # layer.r = ones(layer.nin)
    fill!(layer.r, 1.0)
    copy!(layer.x, x)

    dosigmamult!(layer, layer.p)
    # copy!(layer.Σ, layer.b)
    # BLAS.gemv!('N', layer.p, layer.w, layer.x, 1.0, layer.Σ)
    # layer.Σ = layer.p * (layer.w * layer.x) + layer.b
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
  # layer.δ = multiplyDerivative ? costmult .* backward(layer.activation, layer.Σ) : costmult
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
  # layer.δ = (nextlayer.w' * (nextlayer.nextr .* nextlayer.δ)) .* backward(layer.activation, layer.Σ)
end

# update weights/bias one column at a time... skipping over the dropped out nodes
function updateWeights!(layer::Layer, gradientModel::GradientModel)

  # note: i refers to the output, j refers to the input

  for i in 1:layer.nout

    if layer.nextr[i] > 0.0
      
      # if this node is retained, we can update incoming bias
      bGrad = layer.δ[i]  # δi is the gradient
      # dbi = Δbi(gradientModel, layer.gradientState, bGrad, layer.b[i], i)
      dbi = Δij(gradientModel, layer.dbState, bGrad, layer.b[i], i, 1)
      # Gbi = layer.Gb[i] + δi^2

      # dbi = Gbi > 0.0 ? Δbi(params, δi / (params.useAdagrad ? sqrt(1.0+Gbi) : 1.0), layer.db[i]) : 0.0

      layer.b[i] += dbi
      # layer.db[i] = dbi
      # layer.Gb[i] = Gbi
      
      for j in 1:layer.nin
        
        # if this input node is retained, then we can also update the weight
        if layer.r[j] > 0.0
          
          wGrad = bGrad * layer.x[j]
          # dwij = Δwij(gradientModel, layer.gradientState, wGrad, layer.w[i,j], i, j)
          dwij = Δij(gradientModel, layer.dwState, wGrad, layer.w[i,j], i, j)
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


