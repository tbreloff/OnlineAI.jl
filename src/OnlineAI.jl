
module OnlineAI


using Distributions
using QuickStructs
using OnlineStats
using Qwt
import StatsBase

export Activation,
			 IdentityActivation,
			 SigmoidActivation,
			 TanhActivation,
			 SoftsignActivation,

			 Layer,
			 buildLayer,
			 feedforward!,
			 # update!,

			 NeuralNet,
			 buildNeuralNet,
			 totalerror,

			 SolverData,
			 DataVec,
			 buildSolverData,
			 splitSolverData,
			 # sample,
			 DataSets,

			 SolverParams,
			 SolverStats,
			 buildSolverParams,
			 solve!,

			 buildNet,
			 buildClassifierNet,
			 buildRegressionNet



include("utils.jl")
include("activations.jl")
include("node.jl")
include("layer.jl")
include("net.jl")
include("lstm.jl")
include("data.jl")
include("solver.jl")
include("build.jl")

include("liquid.jl")

end
