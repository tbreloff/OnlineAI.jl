# params should contain all algorithm-specific parameters and methods.
# at a minimum, we need to be able to compute the weight updates for a layer

abstract DropoutStrategy

immutable Dropout <: DropoutStrategy
  # on::Bool 
  pInput::Float64  # the probability that a node is used for the weights from inputs
  pHidden::Float64  # the probability that a node is used for hidden layers
end
Dropout(; pInput=0.8, pHidden=0.5) = Dropout(pInput, pHidden)


immutable NoDropout <: DropoutStrategy end


# -------------------------------------


abstract ErrorModel

"typical sum of squared errors"
immutable L2ErrorModel <: ErrorModel end

"returns M from the equation δ = M .* f'(Σ) ... used in the gradient update"
errorMultiplier(::L2ErrorModel, y::Float64, yhat::Float64) = yhat - y

"cost function"
cost(::L2ErrorModel, y::Float64, yhat::Float64) = 0.5 * (y - yhat) ^ 2
cost(::L2ErrorModel, y::AVecF, yhat::AVecF) = 0.5 * sumabs2(y - yhat)

#-------------

"""
Typical sum of squared errors, but scaled by ρ*y.  We implicitly assume that y ∈ {0,1}.
"""
immutable WeightedL2ErrorModel <: ErrorModel
  ρ::Float64
end
errorMultiplier(model::WeightedL2ErrorModel, y::Float64, yhat::Float64) = (yhat - y) * (y > 0 ? model.ρ : 1)
cost(model::WeightedL2ErrorModel, y::Float64, yhat::Float64) = 0.5 * (y - yhat) ^ 2 * (y > 0 ? model.ρ : 1)

#-------------

immutable CrossEntropyErrorModel <: ErrorModel end

errorMultiplier(model::CrossEntropyErrorModel, y::Float64, yhat::Float64) = yhat - y # binary case
function errorMultiplier(model::CrossEntropyErrorModel, y::AVecF, yhat::AVecF) # softmax case
  (length(y) == 1 ? Float64[errorMultiplier(model, y[1], yhat[1])] : yhat - y), false
end

cost(model::CrossEntropyErrorModel, y::Float64, yhat::Float64) = -log(y > 0.0 ? yhat : (1.0 - yhat)) # binary case
function cost(model::CrossEntropyErrorModel, y::AVecF, yhat::AVecF) # softmax case
  length(y) == 1 && return cost(model, y[1], yhat[1])
  C = 0.0
  for (i,yi) in enumerate(y)
    C -= yi * log(yhat[i])
  end
  C
end


#-------------

# note: the vector version of errorMultiplier also returns a boolean which is true when we
#       need to multiply this value by f'(Σ) when calculating the sensitivities δ
function errorMultiplier(model::ErrorModel, y::AVecF, yhat::AVecF)
  Float64[errorMultiplier(model, y[i], yhat[i]) for i in length(y)], true
end

function cost(model::ErrorModel, y::AVecF, yhat::AVecF)
  sum([cost(model, y[i], yhat[i]) for i in 1:length(y)])
end

# ----------------------------------------

abstract MomentumModel

immutable FixedMomentum <: MomentumModel
  μ::Float64
end
momentum(model::FixedMomentum) = model.μ

type DecayMomentum <: MomentumModel
  μ::Float64
  decayRate::Float64
end
function DecayMomentum(μ_start::Float64, μ_end::Float64, numPeriods::Int)
  decayRate = exp(log(μ_end / μ_start) / numPeriods)
  DecayMomentum(μ_start, decayRate)
end

function momentum(model::DecayMomentum)
  model.μ *= model.decayRate
  model.μ
end

# ----------------------------------------

abstract LearningRateModel

immutable FixedLearningRate <: LearningRateModel
  η::Float64
end
learningRate(model::FixedLearningRate) = model.η

type DecayLearningRate <: LearningRateModel
  η::Float64
  decayRate::Float64
end
function DecayLearningRate(η_start::Float64, η_end::Float64, numPeriods::Int)
  decayRate = exp(log(η_end / η_start) / numPeriods)
  DecayLearningRate(η_start, decayRate)
end

function learningRate(model::DecayLearningRate)
  model.η *= model.decayRate
  model.η
end

# ----------------------------------------

type NetParams{LEARN<:LearningRateModel, MOM<:MomentumModel, DROP<:DropoutStrategy, ERR<:ErrorModel}
  η::LEARN # learning rate
  μ::MOM
  λ::Float64 # L2 penalty term
  dropoutStrategy::DROP
  errorModel::ERR
end

function NetParams(; η=1e-2, μ=0.0, λ=0.0001, dropout=NoDropout(), errorModel=L2ErrorModel())
  η = typeof(η) <: Real ? FixedLearningRate(Float64(η)) : η  # convert numbers to FixedLearningRate
  μ = typeof(μ) <: Real ? FixedMomentum(Float64(μ)) : μ  # convert numbers to FixedMomentum
  NetParams(η, μ, λ, dropout, errorModel)
end

# get the probability that we retain a node using the dropout strategy (returns 1.0 if off)
getDropoutProb(params::NetParams, isinput::Bool) = getDropoutProb(params.dropoutStrategy, isinput)
getDropoutProb(strat::NoDropout, isinput::Bool) = 1.0
getDropoutProb(strat::Dropout, isinput::Bool) = isinput ? strat.pInput : strat.pHidden

# calc update to weight matrix.  TODO: generalize penalty
function ΔW(params::NetParams, gradients::AMatF, w::AMatF, dw::AMatF)
  -learningRate(params.η) * (gradients + params.λ * w) + momentum(params.μ) * dw
end

function ΔWij(params::NetParams, gradient::Float64, wij::Float64, dwij::Float64)
  -learningRate(params.η) * (gradient + params.λ * wij) + momentum(params.μ) * dwij
end

function Δb(params::NetParams, δ::AVecF, db::AVecF)
  -learningRate(params.η) * δ + momentum(params.μ) * db
end

function Δbi(params::NetParams, δi::Float64, dbi::Float64)
  -learningRate(params.η) * δi + momentum(params.μ) * dbi
end
