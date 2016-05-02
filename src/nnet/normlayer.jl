
doc"""
one layer of a multilayer perceptron (neural net)
ninᵢ := noutᵢ₋₁ + 1  (we add the bias term automatically, so there's one extra input)

This is a normalized layer, with inspiration from:
  `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`
    2015 Ioffe and Szegedy
  `http://arxiv.org/pdf/1502.03167.pdf`

forward value := a = f(Σ)
  Σ = yw + b
  y = xhat * β + α
  xhat = (x - μ) / σ

Note: that we solve for w, b, β, and α, and require intermediate sensitivities δΣ and δy
to compute all gradients.

note: w is a parameter for the case of tied weights (it can be a TransposeView!)
"""
type NormalizedLayer{A <: Mapping,
                     MATF <: AbstractMatrix{Float64},
                     GSTATE <: ParameterUpdaterState,
                     WGT <: Weight} <: NeuralNetLayer
  nin::Int
  nout::Int
  activation::A
  ploss::ParameterLoss
  p::Float64  # dropout retention probability

  # the state of the layer

  dwState::Matrix{GSTATE}
  dbState::Matrix{GSTATE}
  dβState::Matrix{GSTATE}
  dαState::Matrix{GSTATE}

  xvar::Vector{Variance{WGT}}  # nin x 1 -- online variances to calculate μ and σ for the normalization step
  w::MATF               # nout x nin -- weights connecting y --> Σ

  x::VecF               # nin x 1 -- input
  xhat::VecF            # nin x 1 -- xhat = standardize(x)
  α::VecF               # nin x 1 --
  y::VecF               # nin x 1 -- y = xhat .* β + α
  δy::VecF              # nin x 1 -- sensitivities for y:  δyᵢ := dC / dyᵢ   (calculated during backward pass)

  b::VecF               # nout x 1 -- bias terms
  Σ::VecF               # nout x 1 -- inner products
  a::VecF               # nout x 1 -- activation := f(Σ)
  δΣ::VecF               # nout x 1 -- sensitivities for Σ:  δΣᵢ := dC / dΣᵢ (calculated during backward pass)

  β::VecF               # nin x 1 -- updated by SGD step, used to compute y
  r::VecF               # nin x 1 -- vector of dropout retention... 0 if we drop this incoming weight, 1 if we keep it
  nextr::VecF           # nout x 1 -- retention of the nodes of this layer (as opposed to r
                        #             which applies to the incoming weights)
end



function NormalizedLayer(nin::Integer, nout::Integer, activation::Mapping,
                         updater::ParameterUpdater, p::Float64 = 1.0;
                         ploss::ParameterLoss = NoParameterLoss(),
                         weightInit::Function = _initialWeights,
                         wgt = ExponentialWeight(500))

  w = weightInit(nin, nout, activation)
  NormalizedLayer(nin, nout, activation, ploss, p,
                  ParameterUpdaterState(updater, nout, nin),
                  ParameterUpdaterState(updater, nout, 1),
                  ParameterUpdaterState(updater, nin, 1),
                  ParameterUpdaterState(updater, nin, 1),
                  [Variance(wgt) for i in 1:nin],
                  w,
                  [zeros(nin) for i in 1:5]...,
                  [zeros(nout) for i in 1:4]...,
                  ones(nin),
                  ones(nin),
                  ones(nout))
end

function Base.print{A,M,G}(io::IO, l::NormalizedLayer{A,M,G})
  print(io, "NormalizedLayer{$(l.nin)=>$(l.nout) $(l.activation) p=$(l.p) ")
  print(io, "‖δΣ‖₁=$(sumabs(l.δΣ)) ‖δy‖₁=$(sumabs(l.δy)) ")
  print(io, "$(M<:TransposeView ? "T" : "")}")
end
function Base.show(io::IO, l::NormalizedLayer)
  print(io, l)
end

function Base.display(::Base.REPL.REPLDisplay, ::MIME"text/plain", l::NormalizedLayer)
  println(l)
  @show l.r
  @show l.nextr
  @show l.x
  println(l.xvar)
  @show l.xhat
  @show l.y
  @show l.Σ
  @show l.a
  @show l.δy
  @show l.δΣ
  @show l.w
  @show l.b
  @show l.β
  @show l.α
  println()
end


# gemv! :: Σ += p * w * x  (note: 'T' would imply p * w' * x)
# note: we generate 2 different functions here, one for w::Matrix, and one for w::TransposeView
@generated function dosigmamult!(layer::NormalizedLayer, α::Float64)
  if layer.parameters[2] <: TransposeView
    return quote
      copy!(layer.Σ, layer.b)
      BLAS.gemv!('T', α, layer.w.mat, layer.y, 1.0, layer.Σ)
    end
  else
    return quote
      copy!(layer.Σ, layer.b)
      BLAS.gemv!('N', α, layer.w, layer.y, 1.0, layer.Σ)
    end
  end
end

# takes input vector, and computes Σⱼ = wⱼ'x + bⱼ  and  Oⱼ = A(Σⱼ)
function forward!(layer::NormalizedLayer, x::AVecF, istraining::Bool)

  # update r and get scale factor p
  if istraining
    # train... randomly drop out incoming nodes
    # note: I said incoming, since this layers weights are the weights connecting the previous layer to this one
    #       So on dropout, we are actually dropping out thr previous layer's nodes...
    for i in 1:layer.nin
      layer.r[i] = rand() <= layer.p ? 1.0 : 0.0
      fit!(layer.xvar[i], x[i])
    end
    p = 1.0
  else
    fill!(layer.r, 1.0)
    p = layer.p
  end

  # update x, xvar, xhat, y
  for i in 1:layer.nin
    layer.x[i] = x[i]

    # update xvar and standardize xi in one call
    layer.xhat[i] = standardize(layer.xvar[i], x[i])

    # we compute y = xhat * β + α, then multiply by r in case we're dropping out
    layer.y[i] = ((layer.xhat[i] * layer.β[i]) + layer.α[i]) * layer.r[i]
  end


  # update Σ
  dosigmamult!(layer, p)

  # forward!(layer.activation, layer.a, layer.Σ)     # activate
  value!(layer.a, layer.activation, layer.Σ)  # activate
end


# function updateSensitivities!(layer::NormalizedLayer, costmult::AVecF, multiplyDerivative::Bool)
#   updateδΣ!(layer, costmult, multiplyDerivative)
#   updateδy!(layer)
# end

function updateSensitivities!(layer::NormalizedLayer, mloss::ModelLoss, output::AVec, target::AVec)
    sensitivity!(layer.δΣ, layer.activation, mloss, layer.Σ, output, target)
    updateδy!(layer)
end

function updateSensitivities!(layer::NormalizedLayer, nextlayer::NormalizedLayer)
  updateδΣ!(layer, nextlayer)
  updateδy!(layer)
end

# # backward step for the final (output) layer
# # note: costmult is the amount to multiply against f'(Σ)... L2 case should be: (yhat-y)
# function updateδΣ!(layer::NormalizedLayer, costmult::AVecF, multiplyDerivative::Bool)
#   copy!(layer.δΣ, costmult)
#   if multiplyDerivative
#     for i in 1:layer.nout
#       layer.δΣ[i] *= backward(layer.activation, layer.Σ[i])
#     end
#   end
# end


# this is the backward step for a hidden layer
# notes: we are figuring out the effect of each node's activation value on the next sensitivities
# TODO: this is unstable... how can we stabilize correctly????
function updateδΣ!(layer::NormalizedLayer, nextlayer::NormalizedLayer)
  for i in 1:layer.nout
    σi = std(nextlayer.xvar[i]) |> if0then1
    βi = nextlayer.β[i]
    d = deriv(layer.activation, layer.Σ[i])
    layer.δΣ[i] = nextlayer.δy[i] * (βi / σi) * d
  end
end

function updateδy!(layer::NormalizedLayer)
  for i in 1:layer.nin
    δyi = 0.0
    for j in 1:layer.nout
      δyi += layer.δΣ[j] * layer.w[j,i]
    end
    layer.δy[i] = δyi
  end
end

doc"""
Update weights/bias one column at a time... skipping over the dropped out nodes.

gradients:
```
  bGrad = δΣ
  wGrad = δΣ * y
  αGrad = δy
  βGrad = δy * xhat
```
"""
function updateWeights!(layer::NormalizedLayer, updater::ParameterUpdater)

    # note: i refers to the output, j refers to the input
    # TODO: change gradient state to operate on one vector only... then have dbState, dwState, etc

    # update w and b
    # note: make sure we only update when the node is retained (from dropout)
    for i in 1:layer.nout
        if layer.nextr[i] > 0.0
            # layer.b[i] += Δij(updater, layer.dbState, layer.δΣ[i], 0.0, i, 1)
            layer.b[i] += param_change!(layer.dbState[i], updater, layer.ploss, layer.δΣ[i], 0.0)

            for j in 1:layer.nin
                if layer.r[j] > 0.0
                    # layer.w[i,j] += Δij(updater, layer.dwState, layer.δΣ[i] * layer.y[j], layer.w[i,j], i, j)
                    layer.w[i,j] += param_change!(layer.dwState[i,j], updater, layer.ploss, layer.δΣ[i] * layer.y[j], layer.w[i,j])
                end
            end
        end
    end


    # update β and α... only when the input is retained (from dropout)
    for i in 1:layer.nin
        if layer.r[i] > 0.0
            # layer.α[i] += Δij(updater, layer.dαState, layer.δy[i], 0.0, i, 1)
            # layer.β[i] += Δij(updater, layer.dβState, layer.δy[i] * layer.xhat[i], 0.0, i, 1)
            layer.α[i] += param_change!(layer.dαState[i], updater, layer.ploss, layer.δy[i], 0.0)
            layer.β[i] += param_change!(layer.dβState[i], updater, layer.ploss, layer.δy[i] * layer.xhat[i], 0.0)
        end
    end


end
