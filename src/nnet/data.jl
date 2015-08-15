

type SolverData
  input::VecF
  target::VecF
end

typealias DataVec Vector{SolverData}

function buildSolverData(inputs::AMatF, targets, indices::AVec{Int} = 1:size(inputs,1))
  @assert size(inputs,1) == size(targets,1)
  sdata = SolverData[]
  for i in indices
    push!(sdata, SolverData(vec(inputs[i,:]), vec(targets[i,:])))
  end
  sdata
end

function splitSolverData(inputs::AMatF, targets, pctValidation::Real, pctTest::Real, randomize::Bool = false)
  n = size(inputs,1)
  indices = collect(1:n)
  if randomize
    indices = shuffle(indices)
  end

  testSize = round(Int, n * pctTest)
  validationSize = round(Int, n * pctValidation)

  trainindices = indices[1:n-testSize-validationSize]
  validindices = indices[n-testSize-validationSize+1:n-testSize]
  testindices = indices[n-testSize+1:n]

  [buildSolverData(inputs, targets, indices) for indices in (trainindices, validindices, testindices)]
end

function Distributions.sample(datalist::DataVec)
  j = abs(rand(Int)) % length(datalist) + 1
  datalist[j]
end

function Distributions.sample(datalist::DataVec, n::Int)
  SolverData[sample(datalist) for i in 1:n]
end

function Base.shuffle(datalist::DataVec)
  indices = collect(1:length(datalist))
  datalist[shuffle(indices)]
end

type DataSets
  trainingSet::DataVec
  validationSet::DataVec
  testSet::DataVec
end

