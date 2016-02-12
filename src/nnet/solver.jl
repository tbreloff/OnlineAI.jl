

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
  plotiter::Int
  plotfields::Vector{Symbol}
  stopepochs::Int
  minerror::Float64
  onbreak::Function
end

function SolverParams(; maxiter = 1000,
                        erroriter = 1000,
                        breakiter = 10000,
                        plotiter = -1,
                        plotfields = Symbol[:x, :xhat, :y, :Σ, :a],
                        stopepochs = 100,
                        minerror = 1e-5,
                        onbreak = donothing) 
  SolverParams(maxiter, erroriter, breakiter, plotiter, plotfields, stopepochs, minerror, onbreak)
end

# ------------------------------------------------------------------------

@enum SolverStatus RUNNING CONVERGED STOPPEDEARLY MAXITER

type SolverStats
  numiter::Int
  trainError::Float64
  validationError::Float64
  bestValidationError::Float64
  epochSinceImprovement::Int
  status::SolverStatus
  bestModel
  plotter
end
SolverStats() = SolverStats(0, Inf, Inf, Inf, 0, RUNNING, nothing, nothing)

Base.string(stats::SolverStats) = "SolverStats{$(stats.status) n=$(stats.numiter), trainerr=$(stats.trainError), valerr=$(stats.validationError), besterr=$(stats.bestValidationError), epochSinceImprovement=$(stats.epochSinceImprovement)}"
Base.print(io::IO, stats::SolverStats) = print(io, string(stats))
Base.show(io::IO, stats::SolverStats) = print(io, string(stats))


# ------------------------------------------------------------------------

doc"""
Batch stochastic gradient descent solver.  Sample from training data to update net, stop if converged
or if validation error is not improving.  Returns SolverStats summary object.
"""
function solve!(net::NetStat, traindata::DataSampler, validationdata::DataSampler, transformY::Bool = false)

  stats = SolverStats()

  # optionally set up the plotter (only if plotiter ≥ 0)
  if net.solverParams.plotiter >= 0
    stats.plotter = NetProgressPlotter(net, stats, net.solverParams.plotfields)
  end

  println("\nsolve: $net\n")

  # loop through maxiter times
  for i in 1:net.solverParams.maxiter

    stats.numiter += 1

    # randomly sample one data item and update the network
    data = sample(traindata)
    fit!(net, data, transformY)

    # update the plot?
    piter = net.solverParams.plotiter
    if (piter > 0 && i % net.solverParams.plotiter == 0) || piter == 0
      fit!(stats.plotter)
    end

    # # check for convergence
    if i % net.solverParams.erroriter == 0
      stats.trainError = totalCost(net, traindata)
      stats.validationError = totalCost(net, validationdata)
      println("Status: $stats")

      # check for improvement in validation error
      if stats.validationError < stats.bestValidationError
        stats.bestValidationError = stats.validationError
        stats.bestModel = copy(net)
        stats.epochSinceImprovement = 0
      else
        stats.epochSinceImprovement += 1

        # early stopping... no improvement
        if stats.epochSinceImprovement >= net.solverParams.stopepochs
          println("Early stopping: $stats")
          stats.status = STOPPEDEARLY
          return stats
        end
      end

      # check if our error is low enough
      if stats.validationError <= net.solverParams.minerror
        println("Converged, breaking: $stats")
        stats.status = CONVERGED
        return stats
      end
    end

    # take a break?
    if i % net.solverParams.breakiter == 0
      net.solverParams.onbreak(net, stats)
    end
  end

  println("Maxiter reached: $stats")
  stats.status = MAXITER
  stats
end


