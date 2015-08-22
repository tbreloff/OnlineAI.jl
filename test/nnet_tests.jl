
module NNetTest

using OnlineAI, FactCheck


function testxor(maxiter::Int; hiddenLayerNodes = [2],
                               hiddenActivation = SigmoidActivation(),
                               finalActivation = SigmoidActivation(),
                               params = NetParams(η=0.3, μ=0.1, λ=1e-5))

  # all xor inputs and results
  inputs = [0 0; 0 1; 1 0; 1 1]
  targets = float(sum(inputs,2) .== 1)

  # all sets are the same
  inputs = inputs .- mean(inputs,1)
  data = DataPoints(inputs, targets)

  # hiddenLayerNodes = [2]
  net = buildRegressionNet(ncols(inputs),
                           ncols(targets),
                           hiddenLayerNodes;
                           hiddenActivation = hiddenActivation,
                           finalActivation = finalActivation,
                           params = params)
  show(net)

  params = SolverParams(maxiter=maxiter, minerror=1e-6)
  solve!(net, params, data, data)

  # output = Float64[predict(net, d.input)[1] for d in data]
  output = vec(predict(net, inputs))
  for (o, d) in zip(output, data)
    println("Result: input=$(d.x) target=$(d.y) output=$o")
  end

  net, output
end


facts("NNet") do

  net, output = testxor(10000, params=NetParams(η=0.3, μ=0.1, λ=1e-5, errorModel=CrossEntropyErrorModel()))
  @fact net --> anything
  @fact output --> roughly([0., 1., 1., 0.], atol=0.03)

  net, output = testxor(10000, hiddenLayerNodes=[2], params=NetParams(η=0.3, μ=0.0, λ=0.0, dropout=OnlineAI.DropoutStrategy(on=true,pInput=1.0,pHidden=0.5)))
  @fact net --> anything
  @fact output --> roughly([0., 1., 1., 0.], atol=0.03)

end # facts


end # module
