
module OnlineAI


using Distributions
using QuickStructs
using OnlineStats
using Qwt
using StatsBase

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
			 buildRegressionNet,

			 GaussianReceptiveField,
			 value,
			 Synapse,
			 DiscreteSynapse,
			 fire!,
			 SpikingNeuron,
			 DiscreteLeakyIntegrateAndFireNeuron,
			 LiquidParams,
			 Liquid,
			 GRFInput,
			 LiquidInput,
			 LiquidStateMachine,
			 liquidState,

			 foreach


# represents a node in an arbitrary graph... typically representing a neuron within a neural net
abstract Node


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
