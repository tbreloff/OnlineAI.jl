

type DataPoint
  x::VecF
  y::VecF
end

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

Distributions.sample(dps::DataPoints) = dps[sample(1:length(dps))]
Distributions.sample(dps::DataPoints, n::Int) = dps[sample(1:length(dps), n)]

Base.shuffle(dps::DataPoints) = DataPoints(shuffle(dps.data))
Base.shuffle!(dps::DataPoints) = shuffle!(dps.data)

# --------------------------------------------------------

"separate a dataset into one DataPoint for each y value"
type DataPartitions
  partitions::Vector{DataPoints}
  numSamples::Int
end

function DataPartitions(dps::DataPoints)
  partitions = DataPoints[]
  for dp in dps
    
    matched = false
    for part in partitions
      if dp.y == part[1].y
        push!(part, dp)
        matched = true
        break
      end
    end
    
    if !matched
      push!(partitions, DataPoints([dp]))
    end
  end

  DataPartitions(partitions, 0)
end

# sample from each y value equally
function Distributions.sample(partitions::DataPartitions)
  parts = partitions.partitions
  dps = parts[(partitions.numSamples%length(parts))+1]
  partitions.numSamples += 1
  sample(dps)
end

Distributions.sample(partitions::DataPartitions, n::Int) = DataPoints([sample(partitions) for i in 1:n])

# --------------------------------------------------------

type DataSets
  trainingSet::DataPoints
  validationSet::DataPoints
  testSet::DataPoints
end

