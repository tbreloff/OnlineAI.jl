
module NNetTest

using OnlineAI, FactCheck

function xor_data()
  inputs = [0 0; 0 1; 1 0; 1 1]
  targets = float(sum(inputs,2) .== 1)

  # all sets are the same
  inputs = inputs .- mean(inputs,1)
  DataPoints(inputs, targets)
end


function testxor(; hiddenLayerNodes = [2],
                   hiddenActivation = SigmoidActivation(),
                   finalActivation = IdentityActivation(),
                   params = NetParams(η=0.3, μ=0.1, λ=1e-5),
                   solverParams = SolverParams(maxiter=10000, minerror=1e-6))

  # all xor inputs and results
  inputs = [0 0; 0 1; 1 0; 1 1]
  targets = float(sum(inputs,2) .== 1)

  # all sets are the same
  inputs = inputs .- mean(inputs,1)
  data = DataPoints(inputs, targets)
  sampler = SimpleSampler(data)

  # hiddenLayerNodes = [2]
  net = buildRegressionNet(ncols(inputs),
                           ncols(targets),
                           hiddenLayerNodes;
                           hiddenActivation = hiddenActivation,
                           finalActivation = finalActivation,
                           params = params)
  show(net)

  solve!(net, solverParams, sampler, sampler)

  # output = Float64[predict(net, d.input)[1] for d in data]
  output = vec(predict(net, inputs))
  for (o, d) in zip(output, data)
    println("Result: input=$(d.x) target=$(d.y) output=$o")
  end

  net, output
end


facts("NNet") do

  minerror = 0.05
  solverParams = SolverParams(maxiter=10000, minerror=minerror*0.8)

  net, output = testxor(params=NetParams(η=0.2, μ=0.0, λ=0.0, costModel=L1CostModel()), solverParams=solverParams)
  @fact net --> anything
  @fact output --> roughly([0., 1., 1., 0.], atol=0.05)

  # net, output = testxor(10000, hiddenLayerNodes=[2], params=NetParams(η=0.3, μ=0.0, λ=0.0, dropout=Dropout(pInput=1.0,pHidden=0.5)))
  # @fact net --> anything
  # @fact output --> roughly([0., 1., 1., 0.], atol=0.03)

end # facts


function test_pretrain(; solve=true, pretr=true, netparams=NetParams(η=0.2), kwargs...)
  data = xor_data()
  sampler = StratifiedSampler(data)

  net = buildRegressionNet(2,1,[2]; params=netparams)
  if pretr
    pretrain(net, sampler; kwargs...)
  end

  if solve
    solve!(net, SolverParams(maxiter=10000), sampler, sampler)
  end

  net, Float64[predict(net,d.x)[1] for d in data]
end


end # module
