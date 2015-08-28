
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
                   solverParams = SolverParams(maxiter=100000),
                   inputTransformer = IdentityTransformer(),
                   doPretrain = true)

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

  if doPretrain
    pretrain(net, sampler, sampler)
  end

  stats = solve!(net, sampler, sampler)

  output = vec(predict(net, inputs))
  for (o, d) in zip(output, data)
    println("Result: input=$(d.x) target=$(d.y) output=$o")
  end

  net, output, stats
end


facts("NNet") do

  atol = 0.05
  solverParams = SolverParams(maxiter=10000, minerror=1e-3)

  net, output, stats = testxor(params=NetParams(gradientModel=SGDModel(η=0.2), costModel=L2CostModel()), solverParams=solverParams, doPretrain=false)
  @fact output --> roughly([0., 1., 1., 0.], atol=atol)

  # net, output, stats = testxor(params=NetParams(gradientModel=AdagradModel(), costModel=L2CostModel()), solverParams=solverParams)
  # @fact output --> roughly([0., 1., 1., 0.], atol=atol)

  # net, output, stats = testxor(params=NetParams(gradientModel=AdadeltaModel(), costModel=CrossEntropyCostModel()), finalActivation=SigmoidActivation(), solverParams=solverParams)
  # @fact output --> roughly([0., 1., 1., 0.], atol=atol)

  # net, output, stats = testxor(params=NetParams(gradientModel=AdadeltaModel(), costModel=CrossEntropyCostModel(), dropout=Dropout(1.0,0.9)),
  #                              finalActivation=SigmoidActivation(), solverParams=solverParams,
  #                              hiddenLayerNodes = [6,6])
  # @fact output --> roughly([0., 1., 1., 0.], atol=atol)

end # facts


function test_pretrain(; solve=true, pretr=true, netparams=NetParams(), kwargs...)
  data = xor_data()
  sampler = StratifiedSampler(data)

  # nin = 1; f = x->x[1:1]
  nin = 2; f = nop
  net = buildRegressionNet(Layer, nin,1,[2]; params=netparams, solverParams=SolverParams(maxiter=10000), inputTransformer=f)
  if pretr
    pretrain(net, sampler, sampler; kwargs...)
  end

  if solve
    solve!(net, sampler, sampler)
  end

  net, Float64[predict(net,d.x)[1] for d in data]
end


end # module
nn = NNetTest
