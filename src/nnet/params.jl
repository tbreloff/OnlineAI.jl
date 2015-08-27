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
getDropoutProb(strat::Dropout, isinput::Bool) = isinput ? strat.pInput : strat.pHidden


immutable NoDropout <: DropoutStrategy end
Base.print(io::IO, d::NoDropout) = print(io, "NoDropout")
Base.show(io::IO, d::NoDropout) = print(io, d)
getDropoutProb(strat::NoDropout, isinput::Bool) = 1.0

# ----------------------------------------

# abstract MomentumModel
# OnlineStats.update!(model::MomentumModel) = nothing

# immutable ConstantMomentum <: MomentumModel
#   μ::Float64
# end
# momentum(model::ConstantMomentum) = model.μ

# Base.print(io::IO, model::ConstantMomentum) = print(io, "const_μ{$(momentum(model))}")
# Base.show(io::IO, model::ConstantMomentum) = print(io, model)

# doc"linear decay from μ_high to μ_low over numPeriods"
# type DecayMomentum <: MomentumModel
#   μ_high::Float64
#   μ_low::Float64
#   numPeriods::Int
#   n::Int
# end
# DecayMomentum(μ_high::Float64, μ_low::Float64, numPeriods::Int) = DecayMomentum(μ_high, μ_low, numPeriods, 0)

# momentum(model::DecayMomentum) = (α = min(model.n / model.numPeriods, 1); model.μ_low * α + model.μ_high * (1-α))
# OnlineStats.update!(model::DecayMomentum) = (model.n += 1; nothing)

# Base.print(io::IO, model::DecayMomentum) = print(io, "decay_μ{$(momentum(model))}")
# Base.show(io::IO, model::DecayMomentum) = print(io, model)

# # ----------------------------------------

# abstract LearningRateModel
# OnlineStats.update!(model::LearningRateModel) = nothing

# immutable ConstantLearningRate <: LearningRateModel
#   η::Float64
# end
# learningRate(model::ConstantLearningRate) = model.η

# Base.print(io::IO, model::ConstantLearningRate) = print(io, "const_η{$(learningRate(model))}")
# Base.show(io::IO, model::ConstantLearningRate) = print(io, model)

# doc"linear decay from η_high to η_low over numPeriods"
# type DecayLearningRate <: LearningRateModel
#   η_high::Float64
#   η_low::Float64
#   numPeriods::Int
#   n::Int
# end
# DecayLearningRate(η_high::Float64, η_low::Float64, numPeriods::Int) = DecayLearningRate(η_high, η_low, numPeriods, 0)

# learningRate(model::DecayLearningRate) = (α = min(model.n / model.numPeriods, 1); model.η_low * α + model.η_high * (1-α))
# OnlineStats.update!(model::DecayLearningRate) = (model.n += 1; nothing)

# Base.print(io::IO, model::DecayLearningRate) = print(io, "decay_η{$(learningRate(model))}")
# Base.show(io::IO, model::DecayLearningRate) = print(io, model)

abstract GradientModel
abstract GradientState

# ----------------------------------------

doc"Stochastic Gradient Descent with Momentum"
immutable SGDModel <: GradientModel
  η::Float64 # learning rate
  μ::Float64 # momentum
  λ::Float64 # L2 penalty term
end
SGDModel(; η=0.1, μ=0.5, λ=1e-5) = SGDModel(η, μ, λ)

immutable SGDState <: GradientState
  dw::MatF
  db::VecF
end
SGDState(nin::Int, nout::Int) = SGDState(zeros(nout,nin), zeros(nout))

getGradientState(model::SGDModel, nin::Int, nout::Int) = SGDState(nin,nout)

function Δwij(model::SGDModel, state::SGDState, gradient::Float64, wij::Float64, i::Int, j::Int)
  dwij = -model.η * (gradient + model.λ * wij) + model.μ * state.dw[i,j]
  state.dw[i,j] = dwij
  dwij
end

function Δbi(model::SGDModel, state::SGDState, gradient::Float64, bi::Float64, i::Int)
  dbi = -model.η * gradient + model.μ * state.db[i]
  state.db[i] = dbi
  dbi
end

# ----------------------------------------

doc"Adaptive Gradient"
immutable AdagradModel <: GradientModel
  ε::Float64  # try 0.01?
  η::Float64 # base learning rate (numerator)
  λ::Float64 # L2 penalty term
end
AdagradModel(; ε=0.01, η=1.0, λ=1e-5) = AdagradModel(ε, η, λ)

type AdagradState <: GradientState
  Gw::MatF
  Gb::VecF
end
AdagradState(nin::Int, nout::Int) = AdagradState(zeros(nout,nin), zeros(nout))

getGradientState(model::AdagradModel, nin::Int, nout::Int) = AdagradState(nin,nout)

function Δwij(model::AdagradModel, state::AdagradState, gradient::Float64, wij::Float64, i::Int, j::Int)
  state.Gw[i,j] += gradient^2
  η = model.η / sqrt(model.ε + state.Gw[i,j])
  -η * (gradient + model.λ * wij)
end

function Δbi(model::AdagradModel, state::AdagradState, gradient::Float64, bi::Float64, i::Int)
  state.Gb[i] += gradient^2
  η = model.η / sqrt(model.ε + state.Gb[i])
  -model.η * gradient
end

# ----------------------------------------

doc"""
See: ADADELTA: An Adaptive Learning Rate Method (Zeiler 2012)

Relatively parameter-free... can probably avoid changing ε and ρ
"""
immutable AdadeltaModel <: GradientModel
  ε::Float64  # try 0.01?
  ρ::Float64  # try 0.97?
  λ::Float64 # L2 penalty term
end
AdadeltaModel(; ε=0.01, ρ=0.97, λ=1e-5) = AdadeltaModel(ε, ρ, λ)

type AdadeltaState <: GradientState
  dwMean::MatF  # exponential avg of w changes (lagged by 1)
  dbMean::VecF  # exponential avg of b changes (lagged by 1)
  GwMean::MatF  # exponential avg of w gradients
  GbMean::VecF  # exponential avg of b gradients
end
AdadeltaState(nin::Int, nout::Int) = AdadeltaState(zeros(nout,nin), zeros(nout), zeros(nout,nin), zeros(nout))

getGradientState(model::AdadeltaModel, nin::Int, nout::Int) = AdadeltaState(nin,nout)

function Δwij(model::AdadeltaModel, state::AdadeltaState, gradient::Float64, wij::Float64, i::Int, j::Int)
  ε, ρ = model.ε, model.ρ

  # average g²
  state.GwMean[i,j] = ρ * state.GwMean[i,j] + (1.0 - ρ) * gradient^2

  # compute learning rate from previous average dw² and current average g²
  η = sqrt(state.dwMean[i,j] + ε) / sqrt(state.GwMean[i,j] + ε)

  # compute change and update average dw²
  dwij = -η * (gradient + model.λ * wij)
  state.dwMean[i,j] = ρ * state.dwMean[i,j] + (1.0 - ρ) * dwij^2
  dwij
end
function Δbi(model::AdadeltaModel, state::AdadeltaState, gradient::Float64, bi::Float64, i::Int)
  ε, ρ = model.ε, model.ρ

  # average g²
  state.GbMean[i] = ρ * state.GbMean[i] + (1.0 - ρ) * gradient^2

  # compute learning rate from previous average db² and current average g²
  η = sqrt(state.dbMean[i] + ε) / sqrt(state.GbMean[i] + ε)

  # compute change and update average db²
  dbi = -η * (gradient + model.λ * bi)
  state.dbMean[i] = ρ * state.dbMean[i] + (1.0 - ρ) * dbi^2
  dbi
end

# ----------------------------------------
# ----------------------------------------

# type NetParams{LEARN<:LearningRateModel, MOM<:MomentumModel, DROP<:DropoutStrategy, ERR<:CostModel}
type NetParams{GRAD<:GradientModel, DROP<:DropoutStrategy, ERR<:CostModel}
  # η::LEARN # learning rate
  # μ::MOM
  # gradientModel::GRAD
  # λ::Float64 # L2 penalty term
  gradientModel::GRAD
  dropoutStrategy::DROP
  costModel::ERR
  # useAdagrad::Bool
end

function NetParams(; gradientModel::GradientModel = AdadeltaModel(),
                     dropout::DropoutStrategy = NoDropout(),
                     costModel::CostModel = L2CostModel())
  NetParams(gradientModel, dropout, costModel)
end

# function NetParams(; η=1.0, μ=0.1, λ=1e-5, dropout=NoDropout(), costModel=L2CostModel(), useAdagrad::Bool = true)
#   η = typeof(η) <: Real ? ConstantLearningRate(Float64(η)) : η  # convert numbers to ConstantLearningRate
#   μ = typeof(μ) <: Real ? ConstantMomentum(Float64(μ)) : μ  # convert numbers to ConstantMomentum
#   NetParams(η, μ, λ, dropout, costModel, useAdagrad)
# end

# Base.print(io::IO, p::NetParams) = print(io, "NetParams{η=$(p.η), μ=$(p.μ), λ=$(p.λ), $(p.dropoutStrategy), $(p.costModel), $(p.useAdagrad ? "Adagrad" : "SGD")}")
Base.print(io::IO, p::NetParams) = print(io, "NetParams{$(p.gradientModel) $(p.dropoutStrategy) $(p.costModel)}")
Base.show(io::IO, p::NetParams) = print(io, p)

# get the probability that we retain a node using the dropout strategy (returns 1.0 if off)
getDropoutProb(params::NetParams, isinput::Bool) = getDropoutProb(params.dropoutStrategy, isinput)

# # calc update to weight matrix.  TODO: generalize penalty
# function ΔW(params::NetParams, gradients::AMatF, w::AMatF, dw::AMatF)
#   -learningRate(params.η) * (gradients + params.λ * w) + momentum(params.μ) * dw
# end

# function Δwij(params::NetParams, gradient::Float64, wij::Float64, dwij::Float64)
#   # -learningRate(params.η) * (gradient + params.λ * wij) + momentum(params.μ) * dwij
#   Δwij(params.gradientModel, params.λ, gradient, wij, dwij)
# end

# # function Δb(params::NetParams, δ::AVecF, db::AVecF)
# #   -learningRate(params.η) * δ + momentum(params.μ) * db
# # end

# function Δbi(params::NetParams, δi::Float64, dbi::Float64)
#   # -learningRate(params.η) * δi + momentum(params.μ) * dbi
#   Δbi(params.gradientModel, params.λ, gradient, wij, dwij)
# end


# OnlineStats.update!(params::NetParams) = (update!(params.η); update!(params.μ))
