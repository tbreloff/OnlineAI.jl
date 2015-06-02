
module OnlineAI

export Activation,
			 IdentityActivation,
			 SigmoidActivation,
			 TanhActivation,
			 SoftsignActivation,

			 Layer,
			 buildLayer,
			 feedforward!,
			 update!,

			 NeuralNet,
			 buildNeuralNet,
			 totalerror,

			 SolverData,
			 DataVec,
			 buildSolverData,
			 splitSolverData,
			 sample,
			 DataSets,

			 SolverParams,
			 SolverStats,
			 buildSolverParams,
			 solve!


include("utils.jl")
include("activations.jl")
include("node.jl")
include("layer.jl")
include("net.jl")
include("lstm.jl")
include("data.jl")
include("solver.jl")

end
