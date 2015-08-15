
module NNetTest

using OnlineAI, FactCheck


function testxor(maxiter::Int)

  # all xor inputs and results
  inputs = float([0 0; 0 1; 1 0; 1 1])
  targets = Float64[(sum(row(inputs,i))==1.0)*0.8+0.1 for i in 1:size(inputs,1)]

  # all sets are the same
  data = buildSolverData(inputs, targets)
  datasets = DataSets(data, data, data)

  net = buildNeuralNet([2,2,1]; η=0.5, μ=0.1)

  params = buildSolverParams(maxiter=maxiter, minerror=1e-6)
  solve!(net, params, datasets)

  for d in data
    output = feedforward!(net, d.input)
    println("Result: input=$(d.input) target=$(d.target) output=$output")
  end

  net
end


facts("NNet") do

  net = testxor(100)
  @fact net --> anything
  
end # facts


end # module
