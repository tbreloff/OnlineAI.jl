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


# ----------------------------------------

abstract MomentumModel

immutable FixedMomentum <: MomentumModel
  μ::Float64
end
momentum(model::FixedMomentum) = model.μ

doc"linear decay from μ_high to μ_low over numPeriods"
type DecayMomentum <: MomentumModel
  μ_high::Float64
  μ_low::Float64
  numPeriods::Int
  n::Int
end
DecayMomentum(μ_high::Float64, μ_low::Float64, numPeriods::Int) = DecayMomentum(μ_high, μ_low, numPeriods, 0)

function momentum(model::DecayMomentum)
  μ = model.μ_low + (model.μ_high - model.μ_low) / min(model.n, model.numPeriods)
  model.n += 1
  μ
end

# ----------------------------------------

abstract LearningRateModel

immutable FixedLearningRate <: LearningRateModel
  η::Float64
end
learningRate(model::FixedLearningRate) = model.η

doc"linear decay from η_high to η_low over numPeriods"
type DecayLearningRate <: MomentumModel
  η_high::Float64
  η_low::Float64
  numPeriods::Int
  n::Int
end
DecayLearningRate(η_high::Float64, η_low::Float64, numPeriods::Int) = DecayLearningRate(η_high, η_low, numPeriods, 0)

function learningRate(model::DecayLearningRate)
  η = model.η_low + (model.η_high - model.η_low) / min(model.n, model.numPeriods)
  model.n += 1
  η
end

# ----------------------------------------

type NetParams{LEARN<:LearningRateModel, MOM<:MomentumModel, DROP<:DropoutStrategy, ERR<:CostModel}
  η::LEARN # learning rate
  μ::MOM
  λ::Float64 # L2 penalty term
  dropoutStrategy::DROP
  costModel::ERR
end

function NetParams(; η=1e-2, μ=0.0, λ=0.0001, dropout=NoDropout(), costModel=L2CostModel())
  η = typeof(η) <: Real ? FixedLearningRate(Float64(η)) : η  # convert numbers to FixedLearningRate
  μ = typeof(μ) <: Real ? FixedMomentum(Float64(μ)) : μ  # convert numbers to FixedMomentum
  NetParams(η, μ, λ, dropout, costModel)
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
