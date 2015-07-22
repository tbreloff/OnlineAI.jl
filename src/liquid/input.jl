
# ---------------------------------------------------------------------

# we use a vector of gaussian receptive fields to map continuous real-valued inputs
# to a series of spike trains

type GaussianReceptiveField
  variance::Variance
  grf_offset::Float64
  grf_width_factor::Float64
end


# create a field that covers a specific area (partially overlapping with the next closest one)
const grf_offsets = [2.0, 6.0, 14.0, 30.0] * 0.5
const grf_width_factors = [1.0, 2.0, 4.0, 8.0] * 0.6
function GaussianReceptiveField(variance::Variance, fieldnum::Int)
  idx = floor(Int, (fieldnum+1.01)/2)
  println("$fieldnum $idx")
  @assert idx > 0 && idx <= length(grf_offsets)
  GaussianReceptiveField(variance, (iseven(fieldnum) ? 1.0 : -1.0) * grf_offsets[idx], grf_width_factors[idx])
end


const STANDARD_NORMAL = Normal()
const CENTER_PDF = pdf(STANDARD_NORMAL, 0.0)
function value(grf::GaussianReceptiveField, x::Float64)
  pdf(STANDARD_NORMAL, (OnlineStats.standardize(grf.variance, x) + grf.grf_offset) / grf.grf_width_factor) / CENTER_PDF
end

OnlineStats.update!(grf::GaussianReceptiveField, x::Float64) = update!(grf.variance, x)


function Qwt.plot(grfs::Vector{GaussianReceptiveField}, rng::FloatIterable)
  y = Float64[value(grfs[i],r) for r in rng, i in 1:length(grfs)]
  plot(rng,y)
end


# ---------------------------------------------------------------------

# # immediately increases the value of u for the postSynaptic neuron
# type ImmediateSynapse <: Synapse
#   postsynapticNeuron::SpikingNeuron
#   weight::Float64
# end

# function fire!(synapse::ImmediateSynapse)
#   synapse.postsynapticNeuron.u += synapse.weight
# end

# ---------------------------------------------------------------------

# # simple neuron to pass along a spike train
# type GRFNeuron <: SpikingNeuron
#   excitatory::Bool
#   u::Float64          # current state
#   ϑ::Float64      # threshold level
#   refractoryPeriodsRemaining::Int  # no activity allowed for this many periods after a spike
#   refractoryPeriodsTotal::Int  # no activity allowed for this many periods after a spike
#   fired::Bool  # did the neuron fire in the most recent period?
#   synapses::Vector{ImmediateSynapse}
# end
# GRFNeuron() = GRFNeuron(true, 0.0, 1.0, 0, 0, false, ImmediateSynapse[])

# weight(n::GRFNeuron) = rand(Uniform(0.1, 1.0)) # TODO: make variable?


# ---------------------------------------------------------------------

type GRFInput <: LiquidInput
  variance::Variance
  grfs::Vector{GaussianReceptiveField}
  neurons::Vector{SpikingNeuron}
end

function GRFInput(liquid, variance::Variance)
  M = 4  # TODO: make variable?
  grfs = GaussianReceptiveField[]
  neurons = SpikingNeuron[]
  for j in 1:M
    push!(grfs, GaussianReceptiveField(variance, j))

    # neuron = GRFNeuron()
    neuron = LIFNeuron(liquid.params)
    postsynapticNeurons = sample(liquid.neurons, liquid.params.pctInput)
    neuron.synapses = [LIFSynapse(psn, weight(neuron)) for psn in postsynapticNeurons]
    push!(neurons, neuron)
  end

  GRFInput(variance, grfs, neurons)
end

# increase neurons' u's by the value of the grf, then fire when threshold crossed
function OnlineStats.update!(input::GRFInput, x::Float64)
  update!(input.variance, x)
  for (j,grf) in enumerate(input.grfs)
    neuron = input.neurons[j]
    neuron.u += value(grf, x)
    neuron.fired = false
    fire!(neuron)
  end
end


# ---------------------------------------------------------------------

function LiquidInputs{W}(liquid, variances::Vector{Variance{W}})
  LiquidInputs([GRFInput(liquid, variance) for variance in variances])
end

# ---------------------------------------------------------------------

