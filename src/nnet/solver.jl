

doc"""
Various parameters for the solve! call.
```
  maxiter := maximum total number of iterations
  erroriter := number of iterations between updating the SolverStats object and checking stopping conditions
  breakiter := number of iterations between the `onbreak` callback
  stopepochs := number of epochs (1 epoch == 1 update to stats) without improvement for early stopping
  minerror := when validation error drops below this, stop
  onbreak := callback function, run once every breakiter iterations. function args: (net::NeuralNet, solverParams::SolverParams, stats::SolverStats)
```
"""
type SolverParams
  maxiter::Int   # maximum total number of iterations
  erroriter::Int  # number of iterations be
  breakiter::Int
  stopepochs::Int
  minerror::Float64
  onbreak::Function
end

function SolverParams(; maxiter=1000, erroriter=1000, breakiter=10000, stopepochs=100, minerror=1e-5, onbreak=donothing) 
  SolverParams(maxiter, erroriter, breakiter, stopepochs, minerror, onbreak)
end

# ------------------------------------------------------------------------


type SolverStats
  numiter::Int
  trainError::Float64
  validationError::Float64
  bestValidationError::Float64
  epochSinceImprovement::Int
  bestModel
end
SolverStats() = SolverStats(0, Inf, Inf, Inf, 0, nothing)

Base.string(stats::SolverStats) = "SolverStats{n=$(stats.numiter), trainerr=$(stats.trainError), valerr=$(stats.validationError), besterr=$(stats.bestValidationError)}"
Base.print(io::IO, stats::SolverStats) = print(io, string(stats))
Base.show(io::IO, stats::SolverStats) = print(io, string(stats))


# ------------------------------------------------------------------------

doc"""
Batch stochastic gradient descent solver.  Sample from training data to update net, stop if converged
or if validation error is not improving.  Returns SolverStats summary object.
"""
function solve!(net::NetStat, solverParams::SolverParams, traindata::DataSampler, validationdata::DataSampler)

  stats = SolverStats()

  # loop through maxiter times
  for i in 1:solverParams.maxiter

    stats.numiter += 1

    # randomly sample one data item and update the network
    data = sample(traindata)
    update!(net, data)

    # # check for convergence
    if i % solverParams.erroriter == 0
      stats.trainError = totalCost(net, traindata)
      stats.validationError = totalCost(net, validationdata)
      println("Status: $stats  $net")

      # check for improvement in validation error
      if stats.validationError < stats.bestValidationError
        stats.bestValidationError = stats.validationError
        stats.bestModel = copy(net)
        stats.epochSinceImprovement = 0
      else
        stats.epochSinceImprovement += 1

        # early stopping... no improvement
        if stats.epochSinceImprovement >= solverParams.stopepochs
          println("Early stopping: $stats")
          return stats
        end
      end

      # check if our error is low enough
      if stats.validationError <= solverParams.minerror
        println("Converged, breaking: $stats")
        return stats
      end
    end

    # take a break?
    if i % solverParams.breakiter == 0
      solverParams.onbreak(net, solverParams, stats)
    end
  end

  println("Maxiter reached: $stats")
  stats
end


