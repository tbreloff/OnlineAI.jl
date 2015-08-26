

# -----------------------------------------------------------------------------

doc"Represents a `Constant Distribution`"
immutable Constant{T} <: DiscreteUnivariateDistribution
  c::T
end
StatsBase.sample(c::Constant) = c.c


# -----------------------------------------------------------------------------


StatsBase.sample(d::Distribution) = rand(d)


doc"""
A `ParameterSampler` holds vectors of keyword parameter names and the distributions to sample from
when randomly choosing a parameter set.  Example:

```
  ps = ParameterSampler([:x, :y], [Constant(1), Normal()])
```
"""
type ParameterSampler
  syms::Vector{Symbol}
  dists::Vector{Distribution}
end
StatsBase.sample(ps::ParameterSampler) = [(ps.syms[i], sample(dist)) for (i,dist) in enumerate(ps.dists)]



# -----------------------------------------------------------------------------


doc"""
Create N models, all using the `buildFn`, which handles creating the model with the constant args/kwargs,
and a random sampling of the parameters from the parameter sampler.
Note we assume all parameters we care about setting are keyword params.
"""
function generateModels{M<:OnlineStat}(buildFn::Function, ps::ParameterSampler, N::Integer)
  OnlineStat[buildFn(; sample(ps)...) for i in 1:N]
end

# -----------------------------------------------------------------------------


# TODO: ensemble tools, parameter optimization (grid search, etc)

doc"""
The Ensemble type represents a set of models which have been fit to the same data.
A call to update! will update all the models with that data.  Then the outputs of 
the models (yhatᵢⱼ) are stacked and get passed through to the `agg` aggregator. `agg`
is another OnlineStat which regresses yhats (model estimates) on y (targets).

Note: predict first generates estimates from each model, then uses those to predict y
with the `agg` model aggregator.
"""
type Ensemble{STAT} <: OnlineStat
  models::Vector{OnlineStat}
  aggs::Vector{STAT}  # one aggregation model for each target variable... each agg maps nummodels vars to numtargets vars
end

Ensemble{M<:OnlineStat, AGG}(buildFn::Function, ps::ParameterSampler, N::Integer, args...; kwargs...) = Ensemble(generateModels(buildFn, ps, N, args...; kwargs...), )


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

# similar to predict, but update! instead
function OnlineStats.update!(ensemble::Ensemble, x::AVecF, y::AVecF)
  for model in ensemble.models
    update!(model, x, y)
  end

  ests = predict(ensemble.models)

  yhat = similar(y)
  for i in 1:length(y)
    esti = Float64[est[i] for est in ests]
    update!(ensemble.aggs[i], esti, y)
  end
end


