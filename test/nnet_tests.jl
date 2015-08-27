
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
                   params = NetParams(),
                   solverParams = SolverParams(maxiter=10000),
                   inputTransformer = IdentityTransformer())

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
                           params = params,
                           solverParams = solverParams,
                           inputTransformer = inputTransformer)
  show(net)

  stats = solve!(net, sampler, sampler)

  output = vec(predict(net, inputs))
  for (o, d) in zip(output, data)
    println("Result: input=$(d.x) target=$(d.y) output=$o")
  end

  net, output, stats
end


facts("NNet") do

  minerror = 0.05
  solverParams = SolverParams(maxiter=10000, minerror=minerror*0.8)

  net, output, stats = testxor(params=NetParams(costModel=L1CostModel()), solverParams=solverParams)
  @fact output --> roughly([0., 1., 1., 0.], atol=0.05)

  net, output, stats = testxor(params=NetParams(gradientModel=AdagradModel(), costModel=L2CostModel()), solverParams=solverParams)
  @fact output --> roughly([0., 1., 1., 0.], atol=0.05)

  net, output, stats = testxor(params=NetParams(gradientModel=AdadeltaModel(), costModel=CrossEntropyCostModel()), finalActivation=SigmoidActivation(), solverParams=solverParams)
  @fact output --> roughly([0., 1., 1., 0.], atol=0.05)

end # facts


function test_pretrain(; solve=true, pretr=true, netparams=NetParams(), kwargs...)
  data = xor_data()
  sampler = StratifiedSampler(data)

  # nin = 1; f = x->x[1:1]
  nin = 2; f = nop
  net = buildRegressionNet(nin,1,[2]; params=netparams, solverParams=SolverParams(maxiter=10000), inputTransformer=f)
  if pretr
    pretrain(net, sampler; kwargs...)
  end

  if solve
    solve!(net, sampler, sampler)
  end

  net, Float64[predict(net,d.x)[1] for d in data]
end


end # module
nn = NNetTest
