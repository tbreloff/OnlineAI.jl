
function buildNet(numInputs::Integer, numOutputs::Integer, hiddenStructure::AVec{Int};
                  hiddenActivation::Activation = SigmoidActivation(),
                  finalActivation::Activation = SigmoidActivation(),
                  params = NetParams(),
                  solverParams = SolverParams(),
                  inputTransformer::Transformer = IdentityTransformer())
  layers = Layer[]
  nin = numInputs

  # the first layer will get the "input" dropout probability
  pDropout = getDropoutProb(params, true)

  for nout in hiddenStructure

    # push the hidden layer
    push!(layers, Layer(nin, nout, hiddenActivation, pDropout))
    nin = nout
    
    # next layers will get the "hidden" dropout probability
    pDropout = getDropoutProb(params, false)
  end

  # push the output layer
  push!(layers, Layer(nin, numOutputs, finalActivation, pDropout))

  NeuralNet(layers, params, solverParams, inputTransformer)
end

buildClassifierNet(args...; kwargs...) = buildNet(args...; kwargs..., finalActivation = SigmoidActivation())
buildRegressionNet(args...; kwargs...) = buildNet(args...; kwargs..., finalActivation = IdentityActivation())