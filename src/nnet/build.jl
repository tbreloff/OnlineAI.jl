
function buildNet(numInputs::Integer, numOutputs::Integer, hiddenStructure::AVec{Int};
                  hiddenActivation::Activation = SigmoidActivation(),
                  finalActivation::Activation = SigmoidActivation(),
                  solver = NNetSolver())
  layers = Layer[]
  nin = numInputs
  for nout in hiddenStructure
    push!(layers, Layer(nin, nout, hiddenActivation))
    nin = nout
  end
  push!(layers, Layer(nin, numOutputs, finalActivation))

  NeuralNet(layers, solver)
end

buildClassifierNet(args...; kwargs...) = buildNet(args...; kwargs..., finalActivation = SigmoidActivation())
buildRegressionNet(args...; kwargs...) = buildNet(args...; kwargs..., finalActivation = IdentityActivation())