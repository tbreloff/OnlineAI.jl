# solver should contain all algorithm-specific parameters and methods.
# at a minimum, we need to be able to compute the weight updates for a layer

type DropoutStrategy
  on::Bool 
  pInput::Float64  # the probability that a node is used for the weights from inputs
  pHidden::Float64  # the probability that a node is used for hidden layers
end

DropoutStrategy(; on=false, pInput=0.8, pHidden=0.5) = DropoutStrategy(on, pInput, pHidden)


# ----------------------------------------

type NNetSolver
  η::Float64 # learning rate
  μ::Float64 # momentum
  λ::Float64 # L2 penalty term
  dropout::DropoutStrategy
end

NNetSolver(; η=1e-2, μ=0.0, λ=0.0001, dropout=DropoutStrategy()) = NNetSolver(η, μ, λ, dropout)

# get the probability that we retain a node using the dropout strategy (returns 1.0 if off)
function getDropoutProb(solver::NNetSolver, isinput::Bool)
  solver.dropout.on ? (isinput ? solver.dropout.pInput : solver.dropout.pHidden) : 1.0
end

# calc update to weight matrix.  TODO: generalize penalty
function ΔW(solver::NNetSolver, gradients::AMatF, w::AMatF, dw::AMatF)
  -solver.η * (gradients + solver.λ * w) + solver.μ * dw
end

function Δb(solver::NNetSolver, δ::AVecF, db::AVecF)
  -solver.η * δ + solver.μ * db
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
totalerror(net::NNetStat, data::DataPoint) = totalerror(net, data.x, data.y)
totalerror(net::NNetStat, dataset::DataPoints) = sum([totalerror(net, data) for data in dataset])


function solve!(net::NNetStat, params::SolverParams, traindata::DataPoints, validationdata::DataPoints)

  stats = SolverStats(0, 0.0)

  # loop through maxiter times
  for i in 1:params.maxiter

    stats.numiter += 1

    # randomly sample one data item and update the network
    data = sample(traindata)
    update!(net, data)

    # # check for convergence
    if i % params.erroriter == 0
      stats.validationError = totalerror(net, validationdata)
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


