
module NNetTest

using OnlineAI, FactCheck


function testxor(maxiter::Int)

  # all xor inputs and results
  inputs = [0 0; 0 1; 1 0; 1 1]
  targets = float(sum(inputs,2) .== 1)

  # all sets are the same
  data = buildSolverData(float(inputs), targets)
  datasets = DataSets(data, data, data)

  hiddenLayerNodes = [2]
  net = buildRegressionNet(ncols(inputs),
                           ncols(targets),
                           hiddenLayerNodes;
                           solver = NNetSolver(η=0.3, μ=0.1, λ=1e-5))
  show(net)

  params = SolverParams(maxiter=maxiter, minerror=1e-6)
  solve!(net, params, datasets)

  # output = Float64[predict(net, d.input)[1] for d in data]
  output = vec(predict(net, float(inputs)))
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
