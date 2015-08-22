# solver should contain all algorithm-specific parameters and methods.
# at a minimum, we need to be able to compute the weight updates for a layer

type DropoutStrategy
  on::Bool 
  pInput::Float64  # the probability that a node is used for the weights from inputs
  pHidden::Float64  # the probability that a node is used for hidden layers
end

DropoutStrategy(; on=false, pInput=0.8, pHidden=0.5) = DropoutStrategy(on, pInput, pHidden)



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


# """
# custom weighted classification error.  has a parameter 0 ≤ ρ ≤ 1 which determines the relative
# importance of sensitivity vs specificity... assumes f(Σ) can take positive and negative values,
# and also assumes that y ∈ {0,1}
# """
# immutable WeightedClassificationErrorModel <: ErrorModel
#   ρ::Float64
# end

# function errorMultiplier(model::WeightedClassificationErrorModel, y::Float64, yhat::Float64)
#   yhat >= 0 ? (1 - model.ρ) * (1 - y) : -model.ρ * y
# end

# function cost(model::WeightedClassificationErrorModel, y::Float64, yhat::Float64)
#   yhat * errorMultiplier(model, y, yhat)
# end


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

type NNetSolver{EM<:ErrorModel}
  η::Float64 # learning rate
  μ::Float64 # momentum
  λ::Float64 # L2 penalty term
  dropout::DropoutStrategy
  errorModel::EM
end

function NNetSolver(; η=1e-2, μ=0.0, λ=0.0001, dropout=DropoutStrategy(), errorModel=L2ErrorModel())
  NNetSolver(η, μ, λ, dropout, errorModel)
end

# get the probability that we retain a node using the dropout strategy (returns 1.0 if off)
function getDropoutProb(solver::NNetSolver, isinput::Bool)
  solver.dropout.on ? (isinput ? solver.dropout.pInput : solver.dropout.pHidden) : 1.0
end

# calc update to weight matrix.  TODO: generalize penalty
function ΔW(solver::NNetSolver, gradients::AMatF, w::AMatF, dw::AMatF)
  -solver.η * (gradients + solver.λ * w) + solver.μ * dw
end

function ΔWij(solver::NNetSolver, gradient::Float64, wij::Float64, dwij::Float64)
  -solver.η * (gradient + solver.λ * wij) + solver.μ * dwij
end

function Δb(solver::NNetSolver, δ::AVecF, db::AVecF)
  -solver.η * δ + solver.μ * db
end

function Δbi(solver::NNetSolver, δi::Float64, dbi::Float64)
  -solver.η * δi + solver.μ * dbi
end


# -------------------------------------


type SolverParams
  maxiter::Int
  erroriter::Int
  minerror::Float64
  displayiter::Int
  onbreak::Function
end

type SolverStats
  numiter::Int
  validationError::Float64
end

function SolverParams(; maxiter=1000, erroriter=1000, minerror=1e-5, displayiter=10000, onbreak=donothing) 
  SolverParams(maxiter, erroriter, minerror, displayiter, onbreak)
end

OnlineStats.update!(net::NNetStat, data::DataPoint) = update!(net, data.x, data.y)


totalCost(net::NNetStat, data::DataPoint) = cost(net, data.x, data.y)
totalCost(net::NNetStat, dataset::DataPoints) = sum([totalCost(net, data) for data in dataset])


function solve!(net::NNetStat, params::SolverParams, traindata::Union(DataPoints,DataPartitions), validationdata::DataPoints)

  stats = SolverStats(0, 0.0)

  # loop through maxiter times
  for i in 1:params.maxiter

    stats.numiter += 1

    # randomly sample one data item and update the network
    data = sample(traindata)
    update!(net, data)

    # # check for convergence
    if i % params.erroriter == 0
      stats.validationError = totalCost(net, validationdata)
      println("Status: iter=$i toterr=$(stats.validationError)  $net")
      if stats.validationError <= params.minerror
        println("Breaking: niter=$i, toterr=$(stats.validationError), minerr=$(params.minerror)")
        return
      end
    end

    if i % params.displayiter == 0
      params.onbreak(net, params, stats)
    end
  end

  println("maxiter reached")
end


