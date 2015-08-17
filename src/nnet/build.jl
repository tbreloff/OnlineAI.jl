
function buildNet(numInputs::Integer, numOutputs::Integer, hiddenStructure::AVec{Int};
                  hiddenActivation::Activation = SigmoidActivation(),
                  finalActivation::Activation = SigmoidActivation(),
                  solver = NNetSolver())
  layers = Layer[]
  nin = numInputs

  # the first layer will get the "input" dropout probability
  pDropout = getDropoutProb(solver, true)

  for nout in hiddenStructure

    # push the hidden layer
    push!(layers, Layer(nin, nout, hiddenActivation, pDropout))
    nin = nout
    
    # next layers will get the "hidden" dropout probability
    pDropout = getDropoutProb(solver, false)
  end

  # push the output layer
  push!(layers, Layer(nin, numOutputs, finalActivation, pDropout))

  NeuralNet(layers, solver)
end

buildClassifierNet(args...; kwargs...) = buildNet(args...; kwargs..., finalActivation = SigmoidActivation())
buildRegressionNet(args...; kwargs...) = buildNet(args...; kwargs..., finalActivation = IdentityActivation())