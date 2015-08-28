
doc"A single input/output pair"
type DataPoint
  x::VecF
  y::VecF
end

# --------------------------------------------------------


# goal: some simple wrapers around datapoint that perform transformations
# we want to: select subsets of values, and transform them (including nop)

doc"""
`Transformation`s wrap an index within a data vector, and descibe how to
calculate Tⱼ(x) by transforming the iᵗʰ index.
"""
abstract Transformation

immutable IdentityTransform <: Transformation idx::Int end
transform(t::IdentityTransform, x::AVec) = x[t.idx]
function transform!(t::IdentityTransform, x_transformed::AVec, x::AVec, i::Int)
  x_transformed[i] = x[t.idx]
end

immutable AbsTransform <: Transformation idx::Int end
transform(t::AbsTransform, x::AVec) = abs(x[t.idx])
function transform!(t::AbsTransform, x_transformed::AVec, x::AVec, i::Int)
  x_transformed[i] = abs(x[t.idx])
end

immutable LogPlus1Transform <: Transformation idx::Int end
transform(t::LogPlus1Transform, x::AVec) = (xi = x[t.idx]; xi > 0.0 ? log(xi+1) : 0.0)
function transform!(t::LogPlus1Transform, x_transformed::AVec, x::AVec, i::Int)
  xi = x[t.idx]
  x_transformed[i] = xi > 0.0 ? log(xi + 1.0) : 0.0
end

immutable SquareTransform <: Transformation idx::Int end
transform(t::SquareTransform, x::AVec) = x[t.idx]^2
function transform!(t::SquareTransform, x_transformed::AVec, x::AVec, i::Int)
  x_transformed[i] = x[t.idx]^2
end

immutable CubeTransform <: Transformation idx::Int end
transform(t::CubeTransform, x::AVec) = x[t.idx]^3
function transform!(t::CubeTransform, x_transformed::AVec, x::AVec, i::Int)
  x_transformed[i] = x[t.idx]^3
end

immutable SignSquareTransform <: Transformation idx::Int end
transform(t::SignSquareTransform, x::AVec) = (xi = x[t.idx]; sign(xi) * xi^2)
function transform!(t::SignSquareTransform, x_transformed::AVec, x::AVec, i::Int)
  xi = x[t.idx]
  x_transformed[i] = sign(xi) * x^2
end


# --------------------------------------------------------

abstract Transformer

immutable IdentityTransformer <: Transformer end
transform(transformer::IdentityTransformer, x) = x
transform!(transformer::IdentityTransformer, x_transformed::AVec, x::AVec) = copy!(x_transformed, x)


immutable VectorTransformer <: Transformer
  transformations::Vector{Transformation}
end
transform{T}(transformer::VectorTransformer, x::AVec{T}) = T[transform(t,x) for t in transformer.transformations]

function transform!(transformer::VectorTransformer, x_transformed::AVec, x::AVec)
  for i in 1:length(x_transformed)
    transform!(transformer.transformations[i], x_transformed, x)
  end
end

# --------------------------------------------------------


doc"A list of input/output data points"
type DataPoints <: AbstractVector{DataPoint}
  data::Vector{DataPoint}
end

function DataPoints(x::AMatF, y::AMatF, indices::AVecI = 1:nrows(x))
  DataPoints([DataPoint(vec(x[i,:]), vec(y[i,:])) for i in indices])
end

function DataPoints(x::AMat, y::AVec, indices::AVecI = 1:nrows(x))
  DataPoints([DataPoint(float(vec(x[i,:])), float(vec(y[i,:]))) for i in indices])
end

function unzip(dps::DataPoints)
  n = length(dps)
  mx = length(first(dps).x)
  my = length(first(dps).y)
  x = zeros(n, mx)
  y = zeros(n, my)
  for (i,dp) in enumerate(dps)
    x[i,:] = dp.x
    y[i,:] = dp.y
  end
  x, y
end


Base.getindex(dps::DataPoints, i::Int) = dps.data[i]
Base.getindex(dps::DataPoints, a::AVecI) = DataPoints(dps.data[a])
Base.setindex!(dps::DataPoints, dp::DataPoint, i::Int) = (dps.data[i] = dp)
Base.push!(dps::DataPoints, dp::DataPoint) = push!(dps.data, dp)
Base.append!(dps::DataPoints, dps2::DataPoints) = append!(dps.data, dps2.data)
Base.length(dps::DataPoints) = length(dps.data)
Base.size(dps::DataPoints) = size(dps.data)

function splitDataPoints(dps::DataPoints, pct::Real)
  r1, r2 = splitRange(length(dps), pct)
  DataPoints[r1], DataPoints[r2]
end

StatsBase.sample(dps::DataPoints) = dps[sample(1:length(dps))]
StatsBase.sample(dps::DataPoints, n::Int) = dps[sample(1:length(dps), n)]

Base.shuffle(dps::DataPoints) = DataPoints(shuffle(dps.data))
Base.shuffle!(dps::DataPoints) = shuffle!(dps.data)


# --------------------------------------------------------

doc"""
Generic approach to sampling data.  Allows for many different approaches in a unified framework.
DataSamplers should implement the StatsBase.sample method which returns a DataPoint object, 
and a DataPoints constructor which returns a DataPoints object with unique DataPoints.
"""
abstract DataSampler

# return a DataPoints object with n samples from the sampler
StatsBase.sample(sampler::DataSampler, n::Int) = DataPoints(DataPoint[sample(sampler) for i in 1:n])

doc"Sample the whole data set"
immutable SimpleSampler <: DataSampler
  data::DataPoints
end
Base.print(io::IO, s::SimpleSampler) = print(io, "SimpleSampler{n=$(length(s.data))}")
Base.show(io::IO, s::SimpleSampler) = print(io,s)

StatsBase.sample(sampler::SimpleSampler) = sample(sampler.data)
DataPoints(sampler::SimpleSampler) = sampler.data

# --------------------------------------------------------

doc"Sample from a subset of the data (defined by the range)"
immutable SubsetSampler <: DataSampler
  data::DataPoints
  range::AVecI
end
Base.print(io::IO, s::SubsetSampler) = print(io, "SubsetSampler{n=$(length(s.data))}")
Base.show(io::IO, s::SubsetSampler) = print(io,s)

# sample from the range
StatsBase.sample(sampler::SubsetSampler) = sampler.data[sample(sampler.range)]

DataPoints(sampler::SubsetSampler) = sampler.data[range]

doc"Create two `SubsetSampler`s, one for the first `pct` of the data and the other for the remaining `1-pct`."
function splitDataSamplers(data::DataPoints, pct::Real)
  r1, r2 = splitRange(length(data), pct)
  SubsetSampler(data, r1), SubsetSampler(data, r2)
end

# --------------------------------------------------------

"separate a dataset into one DataPoint for each y value"
type StratifiedSampler <: DataSampler
  data::DataPoints
  idxlist::Vector{VecI}
  n::Int  # current bucket
end
Base.print(io::IO, s::StratifiedSampler) = print(io, "StratifiedSampler{n=$(length(s.data)), numbuckets=$(length(s.idxlist)), nextbucket=$(s.n)}")
Base.show(io::IO, s::StratifiedSampler) = print(io,s)

function StratifiedSampler(dps::DataPoints)
  idxlist = VecI[]
  for (i,dp) in enumerate(dps)
    
    matched = false
    for indices in idxlist
      if dp.y == dps[first(indices)].y
        push!(indices, i)
        matched = true
        break
      end
    end
    
    if !matched
      push!(idxlist, [i])
    end
  end

  StratifiedSampler(dps, idxlist, 1)
end

# sample from each y value equally
function StatsBase.sample(sampler::StratifiedSampler)

  # get the right bucket
  indices = sampler.idxlist[sampler.n]

  # update the bucket number
  sampler.n += 1
  if sampler.n > length(sampler.idxlist)
    sampler.n = 1 # reset
  end

  # do the sample
  sampler.data[sample(indices)]
end

DataPoints(sampler::StratifiedSampler) = sampler.data


# --------------------------------------------------------

