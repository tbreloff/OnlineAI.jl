

# -----------------------------------------------------------------------------

doc"Represents a `Constant Distribution`"
immutable Constant{T} <: DiscreteUnivariateDistribution
  c::T
end
StatsBase.sample(c::Constant) = c.c


immutable HiddenLayerSampler <: DiscreteMultivariateDistribution
  numlayers::AVecI
  numnodes::AVecI
end
function StatsBase.sample(hls::HiddenLayerSampler)
  numlayers = sample(hls.numlayers)
  Int[sample(hls.numnodes) for i in 1:numlayers]
end


immutable VectorSampler{T} <: DiscreteUnivariateDistribution
  vec::AVec{T}
  probs::VecF
end
VectorSampler(vec::AVec) = VectorSampler(vec, fill(1.0 / length(vec), length(vec)))
VectorSampler{T}(vec::AVec{Tuple{T,Float64}}) = VectorSampler(unzip(vec)...)
function StatsBase.sample(vs::VectorSampler)
  r = rand()
  totp = 0.0
  for (i,p) in enumerate(vs.probs)
    totp += p
    if r <= totp
      return vs.vec[i]
    end
  end
  error("probs don't sum to 1?? $r $totp $vs")
end

# -----------------------------------------------------------------------------


StatsBase.sample(d::Distribution) = rand(d)


doc"""
A `ParameterSampler` holds vectors of keyword parameter names and the distributions to sample from
when randomly choosing a parameter set.  Example:

```
  ps = ParameterSampler(Dict(:x => Constant(1), :y => Normal()))
```
"""
type ParameterSampler
  d::Dict
end
StatsBase.sample(ps::ParameterSampler) = [(s,sample(dist)) for (s,dist) in ps.d]

function Base.print(io::IO, ps::ParameterSampler)
  println("ParameterSampler{")
  for k in sort(keys(ps.d))
    println("  $k => $(typeof(ps.d[k]))")
  end
  print("} ")
end
Base.show(io::IO, ps::ParameterSampler) = print(io, ps)

# -----------------------------------------------------------------------------

doc"""
Creates a `VectorTransformer` by sampling (N times) both a transform type and the index that it maps to.

Note that we call `unique` so as not to duplicate data, so we may actually return less than N samples.
"""
function generateTransformer(transformSampler::VectorSampler{DataType}, indices::AVecI, N::Int)
  VectorTransformer(unique(Transformation[sample(transformSampler)(sample(indices)) for i in 1:N]))
end

doc"""
Create numModels models, all using the `buildFn`, which handles creating the model with the constant args/kwargs,
and a random sampling of the parameters from the parameter sampler.
Note we assume all parameters we care about setting are keyword params.
"""
function generateModels(buildFn::Function, ps::ParameterSampler, numModels::Integer)
  OnlineStat[buildFn(; sample(ps)...) for i in 1:numModels]
end

# -----------------------------------------------------------------------------


# TODO: ensemble tools, parameter optimization (grid search, etc)

doc"""
The Ensemble type represents a set of models which have been fit to the same data.
A call to fit! will update all the models with that data.  Then the outputs of 
the models (yhatᵢⱼ) are stacked and get passed through to the `agg` aggregator. `agg`
is another OnlineStat which regresses yhats (model estimates) on y (targets).

Note: predict first generates estimates from each model, then uses those to predict y
with the `agg` model aggregator.
"""
type Ensemble{STAT} <: OnlineStat
  models::Vector{OnlineStat}
  aggs::Vector{STAT}  # one aggregation model for each target variable... each agg maps nummodels vars to numtargets vars
end

# the build function should take only keyword arguments to build, which can be populated with a ParameterSampler
function Ensemble{AGG<:OnlineStat}(::Type{AGG}, buildFn::Function, ps::ParameterSampler, numModels::Int, nout::Int, args...; kwargs...)
  Ensemble(generateModels(buildFn, ps, numModels), AGG[AGG(args...; kwargs...) for i in 1:nout])
end

# OnlineStats.statenames(o::Ensemble) = [:nobs]
# OnlineStats.state(o::Ensemble) = Any[nobs(o)]
nobs(o::Ensemble) = 0

StatsBase.predict(models::Vector{OnlineStat}, x::AVecF) = [predict(model, x) for model in models]

# get estimates from each model.  then for each i, grab the estimates of the iᵗʰ target variable
# and use those as the inputs to the iᵗʰ aggregation model
function StatsBase.predict(ensemble::Ensemble, x::AVecF)
  ests = predict(ensemble.models)
  nout = length(ensemble.aggs)
  yhat = zeros(nout)
  for i in 1:nout
    esti = Float64[est[i] for est in ests]
    yhat[i] = predict(ensemble.aggs[i], esti)
  end
end

# similar to predict, but fit! instead
function OnlineStats.fit!(ensemble::Ensemble, x::AVecF, y::AVecF)
  for model in ensemble.models
    fit!(model, x, y)
  end

  ests = predict(ensemble.models)

  yhat = similar(y)
  for i in 1:length(y)
    esti = Float64[est[i] for est in ests]
    fit!(ensemble.aggs[i], esti, y)
  end
end


