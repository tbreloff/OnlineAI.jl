

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

update!(nn::NeuralNet, data::SolverData) = update!(nn, data.input, data.target)
totalerror(nn::NeuralNet, data::SolverData) = totalerror(nn, data.input, data.target)
totalerror(nn::NeuralNet, dataset::DataVec) = sum([totalerror(nn, data) for data in dataset])


function solve!(nn::NeuralNet, params::SolverParams, datasets::DataSets)

	stats = SolverStats(0, 0.0)

	# loop through maxiter times
	for i in 1:params.maxiter

		stats.numiter += 1

		# randomly sample one data item and update the network
		data = sample(datasets.trainingSet)
		update!(nn, data)

		# # check for convergence
		if i % params.erroriter == 0
			stats.validationError = totalerror(nn, datasets.validationSet)
			println("Status: iter=$i toterr=$(stats.validationError)")
			if stats.validationError <= params.minerror
				println("Breaking: niter=$i, toterr=$(stats.validationError), minerr=$(params.minerror)")
				return
			end
		end

		if i % params.displayiter == 0
			params.onbreak(nn, params, datasets, stats)
		end
	end

	println("maxiter reached")
end


