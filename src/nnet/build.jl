
function buildNet(numInputs::Integer, numOutputs::Integer, hiddenStructure::AVec{Int};
                  hiddenMapping::Mapping = SigmoidMapping(),
                  finalMapping::Mapping = SigmoidMapping(),
                  params = NetParams(),
                  solverParams = SolverParams(),
                  inputTransformer::Transformer = IdentityTransformer(),
                  wgt::Weight = EqualWeight())
  layers = LAYER[]
  nin = numInputs

  # the first layer will get the "input" dropout probability
  pDropout = getDropoutProb(params, true)

  for nout in hiddenStructure

    # push the hidden layer
    push!(layers, LAYER(nin,
                        nout,
                        hiddenMapping,
                        params.updater,
                        pDropout;
                        wgt = wgt,
                        weightInit = params.weightInit
                       ))
    nin = nout
    
    # next layers will get the "hidden" dropout probability
    pDropout = getDropoutProb(params, false)
  end

  # push the output layer
  push!(layers, LAYER(nin,
                      numOutputs,
                      finalMapping,
                      params.updater,
                      pDropout;
                      wgt = wgt,
                      weightInit = params.weightInit
                     ))

  NeuralNet(layers, params, solverParams, inputTransformer)
end

buildClassificationNet(args...; kwargs...) = buildNet(args...; kwargs..., finalMapping = SigmoidMapping())
buildTanhClassificationNet(args...; kwargs...) = buildNet(args...; kwargs..., hiddenMapping = TanhMapping(), finalMapping = TanhMapping())
buildRegressionNet(args...; kwargs...) = buildNet(args...; kwargs..., finalMapping = IdentityMapping())