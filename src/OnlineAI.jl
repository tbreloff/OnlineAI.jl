
module OnlineAI

using Reexport
using Distributions
@reexport using QuickStructs
@reexport using OnlineStats
@reexport using Qwt
@reexport using CTechCommon
# using StatsBase

import OnlineStats: row, col, row!, col!, rows, cols, nrows, ncols,
                    VecF, MatF, AVec, AMat, AVecF, AMatF

export row, col, row!, col!, rows, cols, nrows, ncols,
       VecF, MatF, AVec, AMat, AVecF, AMatF

export Activation,
       IdentityActivation,
       SigmoidActivation,
       TanhActivation,
       SoftsignActivation,
       ReLUActivation,
       LReLUActivation,

       Layer,
       NeuralNet,

       DataPoint,
       DataPoints,
       splitDataPoints,
       DataPartitions,
       DataSets,

       ErrorModel,
       L2ErrorModel,
       WeightedL2ErrorModel,
       CrossEntropyErrorModel,

       cost,
       totalCost,

       DropoutStrategy,
       Dropout,
       NoDropout,

       NetParams,
       
       SolverParams,
       SolverStats,
       solve!,

       buildNet,
       buildClassifierNet,
       buildRegressionNet


# represents a node in an arbitrary graph... typically representing a neuron within a neural net
abstract NetStat <: OnlineStat
nobs(o::NetStat) = 0
abstract Node


include("utils.jl")
include("nnet/activations.jl")
include("nnet/data.jl")
include("nnet/params.jl")
include("nnet/solver.jl")
include("nnet/layer.jl")
include("nnet/net.jl")
# include("nnet/lstm.jl")
include("nnet/build.jl")
include("nnet/visualize.jl")

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
