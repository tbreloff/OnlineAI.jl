# params should contain all algorithm-specific parameters and methods.
# at a minimum, we need to be able to compute the weight updates for a layer

abstract DropoutStrategy

immutable Dropout <: DropoutStrategy
  # on::Bool 
  pInput::Float64  # the probability that a node is used for the weights from inputs
  pHidden::Float64  # the probability that a node is used for hidden layers
end
Dropout(; pInput=0.8, pHidden=0.5) = Dropout(pInput, pHidden)
Base.print(io::IO, d::Dropout) = print(io, "Dropout{$(d.pInput),$(d.pHidden)}")
Base.show(io::IO, d::Dropout) = print(io, d)


immutable NoDropout <: DropoutStrategy end
Base.print(io::IO, d::NoDropout) = print(io, "NoDropout")
Base.show(io::IO, d::NoDropout) = print(io, d)


# ----------------------------------------

abstract MomentumModel
OnlineStats.update!(model::MomentumModel) = nothing

immutable ConstantMomentum <: MomentumModel
  μ::Float64
end
momentum(model::ConstantMomentum) = model.μ

Base.print(io::IO, model::ConstantMomentum) = print(io, "const_μ{$(momentum(model))}")
Base.show(io::IO, model::ConstantMomentum) = print(io, model)

doc"linear decay from μ_high to μ_low over numPeriods"
type DecayMomentum <: MomentumModel
  μ_high::Float64
  μ_low::Float64
  numPeriods::Int
  n::Int
end
DecayMomentum(μ_high::Float64, μ_low::Float64, numPeriods::Int) = DecayMomentum(μ_high, μ_low, numPeriods, 0)

momentum(model::DecayMomentum) = (α = min(model.n / model.numPeriods, 1); model.μ_low * α + model.μ_high * (1-α))
OnlineStats.update!(model::DecayMomentum) = (model.n += 1; nothing)

Base.print(io::IO, model::DecayMomentum) = print(io, "decay_μ{$(momentum(model))}")
Base.show(io::IO, model::DecayMomentum) = print(io, model)

# ----------------------------------------

abstract LearningRateModel
OnlineStats.update!(model::LearningRateModel) = nothing

immutable ConstantLearningRate <: LearningRateModel
  η::Float64
end
learningRate(model::ConstantLearningRate) = model.η

Base.print(io::IO, model::ConstantLearningRate) = print(io, "const_η{$(learningRate(model))}")
Base.show(io::IO, model::ConstantLearningRate) = print(io, model)

doc"linear decay from η_high to η_low over numPeriods"
type DecayLearningRate <: LearningRateModel
  η_high::Float64
  η_low::Float64
  numPeriods::Int
  n::Int
end
DecayLearningRate(η_high::Float64, η_low::Float64, numPeriods::Int) = DecayLearningRate(η_high, η_low, numPeriods, 0)

learningRate(model::DecayLearningRate) = (α = min(model.n / model.numPeriods, 1); model.η_low * α + model.η_high * (1-α))
OnlineStats.update!(model::DecayLearningRate) = (model.n += 1; nothing)

Base.print(io::IO, model::DecayLearningRate) = print(io, "decay_η{$(learningRate(model))}")
Base.show(io::IO, model::DecayLearningRate) = print(io, model)

# ----------------------------------------

type NetParams{LEARN<:LearningRateModel, MOM<:MomentumModel, DROP<:DropoutStrategy, ERR<:CostModel}
  η::LEARN # learning rate
  μ::MOM
  λ::Float64 # L2 penalty term
  dropoutStrategy::DROP
  costModel::ERR
  useAdagrad::Bool
end

function NetParams(; η=1.0, μ=0.1, λ=1e-5, dropout=NoDropout(), costModel=L2CostModel(), useAdagrad::Bool = true)
  η = typeof(η) <: Real ? ConstantLearningRate(Float64(η)) : η  # convert numbers to ConstantLearningRate
  μ = typeof(μ) <: Real ? ConstantMomentum(Float64(μ)) : μ  # convert numbers to ConstantMomentum
  NetParams(η, μ, λ, dropout, costModel, useAdagrad)
end

Base.print(io::IO, p::NetParams) = print(io, "NetParams{η=$(p.η), μ=$(p.μ), λ=$(p.λ), $(p.dropoutStrategy), $(p.costModel), $(p.useAdagrad ? "Adagrad" : "SGD")}")
Base.show(io::IO, p::NetParams) = print(io, p)

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


OnlineStats.update!(params::NetParams) = (update!(params.η); update!(params.μ))
