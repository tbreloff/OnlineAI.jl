
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

# ----------------------------------------------------------------------

export GaussianReceptiveField,
       value,
       Synapse,
       DelaySynapse,
       fire!,    # checks for threshold crossing, then fires
       SpikingNeuron,
       DiscreteLeakyIntegrateAndFireNeuron,
       LiquidParams,
       Liquid,

       ImmediateSynapse,
       GRFNeuron,
       GRFInput,
       LiquidInput,
       LiquidInputs,

       Readout,
       FireReadout,
       StateReadout,
       FireWindowReadout,

       LiquidStateMachine,
       liquidState,

       visualize,
       LiquidVisualization,
       LiquidVisualizationNode

abstract Synapse
abstract SpikingNeuron <: Node
abstract LiquidInput


include("liquid/readout.jl")
include("liquid/liquid.jl")
include("liquid/neurons.jl")
include("liquid/input.jl")
include("liquid/visualize.jl")


end
