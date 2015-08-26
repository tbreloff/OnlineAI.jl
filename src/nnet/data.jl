
doc"A single input/output pair"
type DataPoint
  x::VecF
  y::VecF
end

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
Base.print(io::IO, s::SimpleSampler) = "SimpleSampler{n=$(length(s.data))}"
Base.show(io::IO, s::SimpleSampler) = print(io,s)

StatsBase.sample(sampler::SimpleSampler) = sample(sampler.data)
DataPoints(sampler::SimpleSampler) = sampler.data

# --------------------------------------------------------

doc"Sample from a subset of the data (defined by the range)"
immutable SubsetSampler <: DataSampler
  data::DataPoints
  range::AVecI
end
Base.print(io::IO, s::SubsetSampler) = "SubsetSampler{n=$(length(s.data))}"
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
Base.print(io::IO, s::StratifiedSampler) = "StratifiedSampler{n=$(length(s.data)), numbuckets=$(length(s.idxlist)), nextbucket=$(s.n)}"
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

