
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

# function Δwij(model::AdagradModel, state::AdagradState, gradient::Float64, wij::Float64, i::Int, j::Int)
#   state.Gw[i,j] += gradient^2
#   η = model.η / sqrt(model.ε + state.Gw[i,j])
#   -η * (gradient + model.λ * wij)
# end

# function Δbi(model::AdagradModel, state::AdagradState, gradient::Float64, bi::Float64, i::Int)
#   state.Gb[i] += gradient^2
#   η = model.η / sqrt(model.ε + state.Gb[i])
#   -model.η * gradient
# end

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

# type AdadeltaState <: GradientState
#   dwMean::MatF  # exponential avg of w changes (lagged by 1)
#   dbMean::VecF  # exponential avg of b changes (lagged by 1)
#   GwMean::MatF  # exponential avg of w gradients
#   GbMean::VecF  # exponential avg of b gradients
# end
# AdadeltaState(nin::Int, nout::Int) = AdadeltaState(zeros(nout,nin), zeros(nout), zeros(nout,nin), zeros(nout))
type AdadeltaState <: GradientState
  dMean::MatF
  GMean::MatF
end

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

# function Δwij(model::AdadeltaModel, state::AdadeltaState, gradient::Float64, wij::Float64, i::Int, j::Int)
#   ε, ρ = model.ε, model.ρ

#   # average g²
#   state.GwMean[i,j] = ρ * state.GwMean[i,j] + (1.0 - ρ) * gradient^2

#   # compute learning rate from previous average dw² and current average g²
#   η = sqrt(state.dwMean[i,j] + ε) / sqrt(state.GwMean[i,j] + ε)

#   # compute change and update average dw²
#   dwij = -η * (gradient + model.λ * wij)
#   state.dwMean[i,j] = ρ * state.dwMean[i,j] + (1.0 - ρ) * dwij^2
#   dwij
# end
# function Δbi(model::AdadeltaModel, state::AdadeltaState, gradient::Float64, bi::Float64, i::Int)
#   ε, ρ = model.ε, model.ρ

#   # average g²
#   state.GbMean[i] = ρ * state.GbMean[i] + (1.0 - ρ) * gradient^2

#   # compute learning rate from previous average db² and current average g²
#   η = sqrt(state.dbMean[i] + ε) / sqrt(state.GbMean[i] + ε)

#   # compute change and update average db²
#   dbi = -η * (gradient + model.λ * bi)
#   state.dbMean[i] = ρ * state.dbMean[i] + (1.0 - ρ) * dbi^2
#   dbi
# end
