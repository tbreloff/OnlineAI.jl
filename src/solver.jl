

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

function buildSolverParams(; maxiter=1000, erroriter=1000, minerror=1e-5, displayiter=10000, onbreak=donothing) 
	SolverParams(maxiter, erroriter, minerror, displayiter, onbreak)
end

OnlineStats.update!(net::NeuralNet, data::SolverData) = update!(net, data.input, data.target)
totalerror(net::NeuralNet, data::SolverData) = totalerror(net, data.input, data.target)
totalerror(net::NeuralNet, dataset::DataVec) = sum([totalerror(net, data) for data in dataset])


function solve!(net::NeuralNet, params::SolverParams, datasets::DataSets)

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


