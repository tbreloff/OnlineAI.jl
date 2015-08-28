
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
AdagradModel(; ε=0.01, η=1.0, λ=1e-5) = AdagradModel(ε, η, λ)

type AdagradState <: GradientState
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
  ρ::Float64  # try 0.97?
  λ::Float64 # L2 penalty term
end
AdadeltaModel(; ε=0.01, ρ=0.97, λ=1e-5) = AdadeltaModel(ε, ρ, λ)


type AdadeltaState <: GradientState
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
  η = sqrt(state.dMean[i,j] + ε) / sqrt(state.GMean[i,j] + ε)

  # compute change and update average dw²
  dij = -η * (gradient + model.λ * val)
  state.dMean[i,j] = ρ * state.dMean[i,j] + (1.0 - ρ) * dij^2
  dij
end

