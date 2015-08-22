

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

OnlineStats.update!(net::NetStat, data::DataPoint) = update!(net, data.x, data.y)


totalCost(net::NetStat, data::DataPoint) = cost(net, data.x, data.y)
totalCost(net::NetStat, dataset::DataPoints) = sum([totalCost(net, data) for data in dataset])


function solve!(net::NetStat, solverParams::SolverParams, traindata::Union(DataPoints,DataPartitions), validationdata::DataPoints)

  stats = SolverStats(0, 0.0)

  # loop through maxiter times
  for i in 1:solverParams.maxiter

    stats.numiter += 1

    # randomly sample one data item and update the network
    data = sample(traindata)
    update!(net, data)

    # # check for convergence
    if i % solverParams.erroriter == 0
      stats.validationError = totalCost(net, validationdata)
      println("Status: iter=$i toterr=$(stats.validationError)  $net")
      if stats.validationError <= solverParams.minerror
        println("Breaking: niter=$i, toterr=$(stats.validationError), minerr=$(solverParams.minerror)")
        return
      end
    end

    if i % solverParams.displayiter == 0
      solverParams.onbreak(net, solverParams, stats)
    end
  end

  println("maxiter reached")
end


