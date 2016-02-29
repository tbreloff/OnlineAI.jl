
# abstract GradientModel
# abstract GradientState

# # ----------------------------------------

"Allows for global storage of a gradient model, so that you don't need to pass it around."
type CurrentUpdater
  updater::ParameterUpdater
end

const _current_gradient_model = CurrentUpdater(AdaMaxModel())

"Get the current global gradient updater used for gradient updates."
current_updater() = _current_gradient_model.updater

"Set the current global gradient updater used for gradient updates."
current_updater!(updater::ParameterUpdater) = (_current_gradient_model.updater = updater)

# "Construct a new `GradientState` object using the model returned from `current_updater()`."
# gradient_state(n::Integer, m::Integer = 1) = gradient_state(current_updater(), n, m)
ParameterUpdaterState(dims::Integer...) = ParameterUpdaterState(current_updater(), dims...)

# # ----------------------------------------

# doc"Stochastic Gradient Descent with Momentum"
# type SGDModel <: GradientModel
#   η::Float64 # learning rate
#   μ::Float64 # momentum
#   λ::Float64 # L2 penalty term
# end
# SGDModel(; η=0.1, μ=0.5, λ=1e-5) = SGDModel(η, μ, λ)

# immutable SGDState <: GradientState
#   lastChanges::MatF
# end
# SGDState(n::Int, m::Int) = SGDState(zeros(n,m))
# gradient_state(model::SGDModel, n::Int, m::Int) = SGDState(n,m)

# # update and return the change
# function Δij(model::SGDModel, state::SGDState, gradient::Float64, val::Float64, i::Int, j::Int)
#   state.lastChanges[i,j] = -model.η * (gradient + model.λ * val) + model.μ * state.lastChanges[i,j]
# end

# # ----------------------------------------

# doc"Adaptive Gradient"
# type AdagradModel <: GradientModel
#   ε::Float64  # try 0.01?
#   η::Float64 # base learning rate (numerator)
#   λ::Float64 # L2 penalty term
# end
# AdagradModel(; ε=1e-8, η=1.0, λ=1e-6) = AdagradModel(ε, η, λ)

# immutable AdagradState <: GradientState
#   G::MatF
# end
# AdagradState(n::Int, m::Int) = AdagradState(zeros(n,m))

# gradient_state(model::AdagradModel, n::Int, m::Int) = AdagradState(n,m)

# function Δij(model::AdagradModel, state::AdagradState, gradient::Float64, val::Float64, i::Int, j::Int)
#   state.G[i,j] += gradient^2
#   η = model.η / sqrt(model.ε + state.G[i,j])
#   -η * (gradient + model.λ * val)
# end

# # ----------------------------------------

# doc"""
# See: ADADELTA: An Adaptive Learning Rate Method (Zeiler 2012)

# Relatively parameter-free... can probably avoid changing ε and ρ
# """
# type AdadeltaModel <: GradientModel
#   ε::Float64  # try 0.01?
#   η::Float64
#   ρ::Float64  # try 0.97?
#   λ::Float64 # L2 penalty term
# end
# AdadeltaModel(; ε=1e-8, η=0.1, ρ=0.95, λ=1e-6) = AdadeltaModel(ε, η, ρ, λ)


# immutable AdadeltaState <: GradientState
#   dMean::MatF
#   GMean::MatF
# end
# AdadeltaState(n::Int, m::Int) = AdadeltaState(zeros(n,m), zeros(n,m))
# gradient_state(model::AdadeltaModel, n::Int, m::Int) = AdadeltaState(n,m)

# function Δij(model::AdadeltaModel, state::AdadeltaState, gradient::Float64, val::Float64, i::Int, j::Int)
#   ε, ρ = model.ε, model.ρ

#   # average g²
#   state.GMean[i,j] = ρ * state.GMean[i,j] + (1.0 - ρ) * gradient^2

#   # compute learning rate from previous average dw² and current average g²
#   η = model.η * sqrt(state.dMean[i,j] + ε) / sqrt(state.GMean[i,j] + ε)

#   # compute change and update average dw²
#   dij = -η * (gradient + model.λ * val)
#   state.dMean[i,j] = ρ * state.dMean[i,j] + (1.0 - ρ) * dij^2
#   dij
# end

# """
# see: ADAM: A method for Stochastic Optimization (Kingma and Ba 2015)

# Tracks an exponential moving average of the first and second moments of the gradient,
# adjusting for zero-bias.  The defaults are those suggested in the paper.

# TODO: AdaMax is similar, using the p-norm as p -> ∞
# """
# type AdamModel <: GradientModel
#   ε::Float64  # small number so we don't divide by 0
#   η::Float64  # learning rate... (this is α in the paper) maybe use around 1e-3?
#   ρ1::Float64 # decay for first moment (β₁ in the paper)
#   ρ2::Float64 # decay for second moment (β₂ in the paper)
#   λ::Float64  # L2 penalty term
# end
# AdamModel(; ε=1e-8, η=1e-3, ρ1=0.9, ρ2=0.999, λ=1e-6) = AdamModel(ε, η, ρ1, ρ2, λ)

# type AdamState <: GradientState
#   m::MatF # average first moment
#   v::MatF # average second moment
#   ρ1t::Float64  # β₁ᵗ from the paper... t-th power of β₁
#   ρ2t::Float64  # β₂ᵗ from the paper... t-th power of β₂
# end
# AdamState(n::Integer, m::Integer) = AdamState(zeros(n,m), zeros(n,m), 1.0, 1.0)
# gradient_state(model::AdamModel, n::Integer, m::Integer) = AdamState(n,m)

# function Δij(model::AdamModel, state::AdamState, gradient::Float64, val::Float64, i::Int, j::Int)
#   ρ1, ρ2 = model.ρ1, model.ρ2
#   state.m[i,j] = ρ1 * state.m[i,j] + (1.0 - ρ1) * gradient
#   state.v[i,j] = ρ2 * state.v[i,j] + (1.0 - ρ2) * gradient^2
#   if i == 1 && j == 1
#     state.ρ1t *= model.ρ1
#     state.ρ2t *= model.ρ2
#   end
#   ηt = model.η * (sqrt(1.0 - state.ρ2t) / (1.0 - state.ρ1t))
#   -ηt * state.m[i,j] / (sqrt(state.v[i,j] + model.ε))
# end


# """
# see: ADAM: A method for Stochastic Optimization (Kingma and Ba 2015)

# AdaMax is similar to Adam, using the p-norm as p -> ∞
# """
# type AdaMaxModel <: GradientModel
#   # ε::Float64  # small number so we don't divide by 0
#   η::Float64  # learning rate... (this is α in the paper) maybe use around 1e-3?
#   ρ1::Float64 # decay for first moment (β₁ in the paper)
#   ρ2::Float64 # decay for second moment (β₂ in the paper)
#   λ::Float64  # L2 penalty term
# end
# AdaMaxModel(; η=1e-3, ρ1=0.9, ρ2=0.99, λ=1e-6) = AdaMaxModel(η, ρ1, ρ2, λ)

# immutable AdaMaxState <: GradientState
#   m::MatF # average first moment
#   u::MatF # average second moment
#   ρ1t::Vector{Float64}  # β₁ᵗ from the paper... t-th power of β₁
#   # ρ2t::Float64  # β₂ᵗ from the paper... t-th power of β₂
# end
# AdaMaxState(n::Integer, m::Integer) = AdaMaxState(zeros(n,m), zeros(n,m), [1.0])
# gradient_state(model::AdaMaxModel, n::Integer, m::Integer) = AdaMaxState(n,m)

# function Δij(model::AdaMaxModel, state::AdaMaxState, gradient::Float64, val::Float64, i::Int, j::Int)
#   # ρ1, ρ2 = model.ρ1, model.ρ2
#   ρ1 = model.ρ1
#   # state.m[i,j] = ρ1 * state.m[i,j] + (1.0 - ρ1) * gradient
#   mij = ρ1 * state.m[i,j] + (1.0 - ρ1) * gradient
#   state.m[i,j] = mij
#   # state.u[i,j] = ρ2 * state.u[i,j] + (1.0 - ρ2) * gradient^2
#   uij = max(model.ρ2 * state.u[i,j], abs(gradient))
#   state.u[i,j] = uij
#   if i == 1 && j == 1
#     state.ρ1t[1] *= ρ1
#   end
#   # ηt = model.η / (1.0 - state.ρ1t[1])
#   # -ηt * mij / (uij + 1e-10)
#   -model.η * mij / ((uij + 1e-10) * (1.0 - state.ρ1t[1]))
# end

# --------------------------------------------------------------

"Enacts a strategy to adjust the learning rate"
abstract LearningRateModel

immutable FixedLearningRate <: LearningRateModel end
OnlineStats.fit!(lrmodel::FixedLearningRate, err::Float64) = nothing


"Adapts learning rate based on relative variance of the changes in the test error"
immutable AdaptiveLearningRate <: LearningRateModel
  gradientModel::GradientModel
  errordiff::Diff
  diffvar::Variance
  adjustmentPct::Float64
  cutoffRatio::Float64
end

function AdaptiveLearningRate(gradientModel::GradientModel,
                              adjustmentPct = 1e-2,
                              cutoffRatio = 1e-1;
                              wgt = ExponentialWeight(20))
  AdaptiveLearningRate(gradientModel, Diff(), Variance(wgt), adjustmentPct, cutoffRatio)
end

# if the error is decreasing at a large rate relative to the variance, increase the learning rate (speed it up)
function OnlineStats.fit!(lrmodel::AdaptiveLearningRate, err::Float64)
  fit!(lrmodel.errordiff, err)
  fit!(lrmodel.diffvar, diff(lrmodel.errordiff))
  m = mean(lrmodel.diffvar)
  s = std(lrmodel.diffvar)
  if s > 0.0
    pct = lrmodel.adjustmentPct * (m / s < -lrmodel.cutoffRatio ? 1.0 : -1.0)
    lrmodel.gradientModel.η *= (1.0 + pct)
  end
end

