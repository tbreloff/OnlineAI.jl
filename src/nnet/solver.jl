# solver should contain all algorithm-specific parameters and methods.
# at a minimum, we need to be able to compute the weight updates for a layer

type NNetSolver
  η::Float64 # learning rate
  μ::Float64 # momentum
  λ::Float64 # L2 penalty term
end

NNetSolver(; η=1e-2, μ=0.0, λ=0.0001) = NNetSolver(η, μ, λ)

# calc update to weight matrix.  TODO: generalize penalty
function ΔW(solver::NNetSolver, gradients::AMatF, w::AMatF, dw::AMatF)
  # map(x->println(size(x)), Any[gradients, w, dw])
  -solver.η * (gradients + solver.λ * w) + solver.μ * dw
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

OnlineStats.update!(net::NNetStat, data::SolverData) = update!(net, data.input, data.target)
totalerror(net::NNetStat, data::SolverData) = totalerror(net, data.input, data.target)
totalerror(net::NNetStat, dataset::DataVec) = sum([totalerror(net, data) for data in dataset])


function solve!(net::NNetStat, params::SolverParams, datasets::DataSets)

  stats = SolverStats(0, 0.0)

  # loop through maxiter times
  for i in 1:params.maxiter

    stats.numiter += 1

    # randomly sample one data item and update the network
    data = sample(datasets.trainingSet)
    update!(net, data)

    # # check for convergence
    if i % params.erroriter == 0
      stats.validationError = totalerror(net, datasets.validationSet)
      println("Status: iter=$i toterr=$(stats.validationError)")
      if stats.validationError <= params.minerror
        println("Breaking: niter=$i, toterr=$(stats.validationError), minerr=$(params.minerror)")
        return
      end
    end

    if i % params.displayiter == 0
      params.onbreak(net, params, datasets, stats)
    end
  end

  println("maxiter reached")
end


