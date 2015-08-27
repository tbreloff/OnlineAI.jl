
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



# represents a node in an arbitrary graph... typically representing a neuron within a neural net
abstract NetStat <: OnlineStat
nobs(o::NetStat) = 0
abstract Node

# ------------------------------------------------

include("utils.jl")

export 
  Activation,
  IdentityActivation,
  SigmoidActivation,
  TanhActivation,
  SoftsignActivation,
  ReLUActivation,
  LReLUActivation,
  SoftmaxActivation
include("nnet/activations.jl")

export
  DataPoint,
  Transformation,
  IdentityTransform,
  AbsTransform,
  LogPlus1Transform,
  SquareTransform,
  CubeTransform,
  SignSquareTransform,
  Transformer,
  IdentityTransformer,
  VectorTransformer,
  transform,
  DataPoints,
  splitDataPoints,
  DataSampler,
  SimpleSampler,
  SubsetSampler,
  splitDataSamplers,
  StratifiedSampler
include("nnet/data.jl")

export
  CostModel,
  L2CostModel,
  L1CostModel,
  WeightedL2CostModel,
  CrossEntropyCostModel,
  cost,
  totalCost
include("nnet/costs.jl")

export
  DropoutStrategy,
  Dropout,
  NoDropout,
  MomentumModel,
  ConstantMomentum,
  DecayMomentum,
  momentum,
  LearningRateModel,
  ConstantLearningRate,
  DecayLearningRate,
  learningRate,
  NetParams
include("nnet/params.jl")

export
  SolverParams,
  SolverStats,
  solve!
include("nnet/solver.jl")

export
  Layer
include("nnet/layer.jl")

export
  NeuralNet
include("nnet/net.jl")

export
  pretrain
include("nnet/pretrain.jl")

# include("nnet/lstm.jl")

export
  buildNet,
  buildClassifierNet,
  buildRegressionNet
include("nnet/build.jl")

export
  Constant,
  HiddenLayerSampler,
  VectorSampler,
  ParameterSampler,
  generateTransformer,
  generateModels,
  Ensemble
include("nnet/ensembles.jl")

export
  visualize
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

# ------------------------------------------------

end
