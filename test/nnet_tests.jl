
module NNetTest

using OnlineAI, FactCheck
const AI = OnlineAI


function testxor(maxiter::Int)

  # all xor inputs and results
  inputs = float([0 0; 0 1; 1 0; 1 1])
  # targets = Float64[(sum(AI.row(inputs,i))==1.0)*0.8+0.1 for i in 1:size(inputs,1)]
  targets = Float64[sum(AI.row(inputs,i))==1.0 for i in 1:nrows(inputs)]

  # all sets are the same
  data = buildSolverData(inputs, targets)
  datasets = DataSets(data, data, data)

  net = NeuralNet([2,2,1]; solver = NNetSolver(η=0.3, μ=0.1, λ=0.0))

  params = SolverParams(maxiter=maxiter, minerror=1e-6)
  solve!(net, params, datasets)

  output = Float64[forward(net, d.input)[1] for d in data]
  for (o, d) in zip(output, data)
    println("Result: input=$(d.input) target=$(d.target) output=$o")
  end

  net, output
end


facts("NNet") do

  net, output = testxor(10000)
  @fact net --> anything
  @fact output --> roughly([0., 1., 1., 0.], atol=0.03)
  
end # facts


end # module
