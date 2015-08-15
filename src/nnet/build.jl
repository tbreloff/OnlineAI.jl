
function buildNet(numInputs::Int, numOutputs::Int, hiddenStructure::VecI;
                  hiddenActivation::Activation = SigmoidActivation(),
                  finalActivation::Activation = SigmoidActivation(),
                  η::Float64 = 0.02,
                  μ::Float64 = 0.2)
  layers = Layer[]
  nin = numInputs
  for nout in hiddenStructure
    push!(layers, buildLayer(nin, nout, hiddenActivation))
    nin = nout
  end
  push!(layers, buildLayer(nin, numOutputs, finalActivation))

  NeuralNet(layers, η, μ)
end

buildClassifierNet(args...; kwargs...) = buildNet(args...; kwargs..., finalActivation = SigmoidActivation())
buildRegressionNet(args...; kwargs...) = buildNet(args...; kwargs..., finalActivation = IdentityActivation())