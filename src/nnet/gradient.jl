
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
  lastChanges::MatF
end
SGDState(n::Int, m::Int) = SGDState(zeros(n,m))
getGradientState(model::SGDModel, n::Int, m::Int) = SGDState(n,m)

# update and return the change
function Δij(model::SGDModel, state::SGDState, gradient::Float64, val::Float64, i::Int, j::Int)
  state.lastChanges[i,j] = -model.η * (gradient + model.λ * val) + model.μ * state.lastChanges[i,j]
end

# ----------------------------------------

doc"Adaptive Gradient"
immutable AdagradModel <: GradientModel
  ε::Float64  # try 0.01?
  η::Float64 # base learning rate (numerator)
  λ::Float64 # L2 penalty term
end
AdagradModel(; ε=1e-8, η=1.0, λ=1e-6) = AdagradModel(ε, η, λ)

immutable AdagradState <: GradientState
  G::MatF
end
AdagradState(n::Int, m::Int) = AdagradState(zeros(n,m))

getGradientState(model::AdagradModel, n::Int, m::Int) = AdagradState(n,m)

function Δij(model::AdagradModel, state::AdagradState, gradient::Float64, val::Float64, i::Int, j::Int)
  state.G[i,j] += gradient^2
  η = model.η / sqrt(model.ε + state.G[i,j])
  -η * (gradient + model.λ * val)
end

# ----------------------------------------

doc"""
See: ADADELTA: An Adaptive Learning Rate Method (Zeiler 2012)

Relatively parameter-free... can probably avoid changing ε and ρ
"""
immutable AdadeltaModel <: GradientModel
  ε::Float64  # try 0.01?
  η::Float64
  ρ::Float64  # try 0.97?
  λ::Float64 # L2 penalty term
end
AdadeltaModel(; ε=1e-8, η=0.1, ρ=0.95, λ=1e-6) = AdadeltaModel(ε, η, ρ, λ)


immutable AdadeltaState <: GradientState
  dMean::MatF
  GMean::MatF
end
AdadeltaState(n::Int, m::Int) = AdadeltaState(zeros(n,m), zeros(n,m))
getGradientState(model::AdadeltaModel, n::Int, m::Int) = AdadeltaState(n,m)

function Δij(model::AdadeltaModel, state::AdadeltaState, gradient::Float64, val::Float64, i::Int, j::Int)
  ε, ρ = model.ε, model.ρ

  # average g²
  state.GMean[i,j] = ρ * state.GMean[i,j] + (1.0 - ρ) * gradient^2

  # compute learning rate from previous average dw² and current average g²
  η = model.η * sqrt(state.dMean[i,j] + ε) / sqrt(state.GMean[i,j] + ε)

  # compute change and update average dw²
  dij = -η * (gradient + model.λ * val)
  state.dMean[i,j] = ρ * state.dMean[i,j] + (1.0 - ρ) * dij^2
  dij
end

"""
see: ADAM: A method for Stochastic Optimization (Kingma and Ba 2015)

Tracks an exponential moving average of the first and second moments of the gradient,
adjusting for zero-bias.  The defaults are those suggested in the paper.

TODO: AdaMax is similar, using the p-norm as p -> ∞
"""
immutable AdamModel <: GradientModel
  ε::Float64  # small number so we don't divide by 0
  η::Float64  # learning rate... (this is α in the paper) maybe use around 1e-3?
  ρ1::Float64 # decay for first moment (β₁ in the paper)
  ρ2::Float64 # decay for second moment (β₂ in the paper)
  λ::Float64  # L2 penalty term
end
AdamModel(; ε=1e-8, η=1e-3, ρ1=0.9, ρ2=0.999, λ=1e-6) = AdamModel(ε, η, ρ1, ρ2, λ)

type AdamState <: GradientState
  m::MatF # average first moment
  v::MatF # average second moment
  ρ1t::Float64  # β₁ᵗ from the paper... t-th power of β₁
  ρ2t::Float64  # β₂ᵗ from the paper... t-th power of β₂
end
AdamState(n::Integer, m::Integer) = AdamState(zeros(n,m), zeros(n,m), 1.0, 1.0)
getGradientState(model::AdamModel, n::Integer, m::Integer) = AdamState(n,m)

function Δij(model::AdamModel, state::AdamState, gradient::Float64, val::Float64, i::Int, j::Int)
  ρ1, ρ2 = model.ρ1, model.ρ2
  state.m[i,j] = ρ1 * state.m[i,j] + (1.0 - ρ1) * gradient
  state.v[i,j] = ρ2 * state.v[i,j] + (1.0 - ρ2) * gradient^2
  state.ρ1t *= model.ρ1
  state.ρ2t *= model.ρ2
  ηt = model.η * (sqrt(1.0 - state.ρ2t) / (1.0 - state.ρ1t))
  -ηt * state.m[i,j] / (sqrt(state.v[i,j] + model.ε))
end


