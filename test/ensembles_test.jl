
module EnsemblesTest

using OnlineAI, FactCheck, Distributions

function xor_data()
  inputs = [0 0; 0 1; 1 0; 1 1]
  targets = float(sum(inputs,2) .== 1)

  # all sets are the same
  inputs = inputs .- mean(inputs,1)
  DataPoints(inputs, targets)
end


# function testxor(; hiddenLayerNodes = [2],
#                    hiddenMapping = SigmoidMapping(),
#                    finalMapping = IdentityMapping(),
#                    params = NetParams(),
#                    solverParams = SolverParams(maxiter=10000))

#   # all xor inputs and results
#   inputs = [0 0; 0 1; 1 0; 1 1]
#   targets = float(sum(inputs,2) .== 1)

#   # all sets are the same
#   inputs = inputs .- mean(inputs,1)
#   data = DataPoints(inputs, targets)
#   sampler = SimpleSampler(data)

#   # hiddenLayerNodes = [2]
#   net = buildRegressionNet(ncols(inputs),
#                            ncols(targets),
#                            hiddenLayerNodes;
#                            hiddenMapping = hiddenMapping,
#                            finalMapping = finalMapping,
#                            params = params)
#   show(net)

#   stats = solve!(net, solverParams, sampler, sampler)

#   # output = Float64[predict(net, d.input)[1] for d in data]
#   output = vec(predict(net, inputs))
#   for (o, d) in zip(output, data)
#     println("Result: input=$(d.x) target=$(d.y) output=$o")
#   end

#   net, output, stats
# end


facts("Ensembles") do

  # minerror = 0.05
  # solverParams = SolverParams(maxiter=10000, minerror=minerror*0.8)

  # net, output, stats = testxor(params=NetParams(μ=0.0, λ=0.0, mloss=L1CostModel()), solverParams=solverParams)
  # @fact net --> anything
  # @fact output --> roughly([0., 1., 1., 0.], atol=0.05)

  ps = ParameterSampler(Dict(zip([:x, :y, :z, :a], [Constant(1), Normal(), Uniform(3,4), VectorSampler(2:4, [0.8,0.1,0.1])])))
  show(ps.d)
  for i in 1:20
    println(sample(ps))
  end
  f(;kwargs...) = println(Dict(kwargs))
  f(;sample(ps)...)

end # facts



end # module
