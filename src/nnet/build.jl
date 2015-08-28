
function buildNet{T<:NeuralNetLayer}(numInputs::Integer, numOutputs::Integer, hiddenStructure::AVec{Int};
                                    hiddenActivation::Activation = SigmoidActivation(),
                                    finalActivation::Activation = SigmoidActivation(),
                                    params = NetParams(),
                                    solverParams = SolverParams(),
                                    inputTransformer::Transformer = IdentityTransformer(),
                                    layerType::Type{T} = Layer)
  layers = NeuralNetLayer[]
  nin = numInputs

  # the first layer will get the "input" dropout probability
  pDropout = getDropoutProb(params, true)

  for nout in hiddenStructure

    # push the hidden layer
    push!(layers, layerType(nin, nout, hiddenActivation, params.gradientModel, pDropout))
    nin = nout
    
    # next layers will get the "hidden" dropout probability
    pDropout = getDropoutProb(params, false)
  end

  # push the output layer
  push!(layers, layerType(nin, numOutputs, finalActivation, params.gradientModel, pDropout))

  NeuralNet(layers, params, solverParams, inputTransformer)
end

buildClassificationNet(args...; kwargs...) = buildNet(args...; kwargs..., finalActivation = SigmoidActivation())
buildRegressionNet(args...; kwargs...) = buildNet(args...; kwargs..., finalActivation = IdentityActivation())