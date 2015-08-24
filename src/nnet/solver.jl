
type SolverParams
  maxiter::Int
  erroriter::Int
  minerror::Float64
  displayiter::Int
  maxEpochWithoutImprovement::Int
  onbreak::Function
end

function SolverParams(; maxiter=1000, erroriter=1000, minerror=1e-5, displayiter=10000, maxEpochWithoutImprovement=100, onbreak=donothing) 
  SolverParams(maxiter, erroriter, minerror, displayiter, maxEpochWithoutImprovement, onbreak)
end

# ------------------------------------------------------------------------


type SolverStats
  numiter::Int
  trainError::Float64
  validationError::Float64
  bestValidationError::Float64
  epochSinceImprovement::Int
end
SolverStats() = SolverStats(0, Inf, Inf, Inf, 0)

Base.string(stats::SolverStats) = "SolverStats{n=$(stats.numiter), trainerr=$(stats.trainError), valerr=$(stats.validationError), besterr=$(stats.bestValidationError)}"
Base.print(io::IO, stats::SolverStats) = print(io, string(stats))
Base.show(io::IO, stats::SolverStats) = print(io, string(stats))


# ------------------------------------------------------------------------

function solve!(net::NetStat, solverParams::SolverParams, traindata::DataSampler, validationdata::DataSampler)

  stats = SolverStats(0, 0.0)

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

      if stats.validationError <= solverParams.minerror
        println("Converged, breaking: $stats")
        return
      end

      # check for improvement in validation error
      if stats.validationError < stats.bestValidationError
        stats.bestValidationError = stats.validationError
        stats.epochSinceImprovement = 0
      else
        stats.epochSinceImprovement += 1

        # early stopping... no improvement
        if stats.epochSinceImprovement >= solverParams.maxEpochWithoutImprovement
          println("Early stopping: $stats")
          return
        end

      end
    end

    if i % solverParams.displayiter == 0
      solverParams.onbreak(net, solverParams, stats)
    end
  end

  println("Maxiter reached: $stats")
end


