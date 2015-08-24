



doc"""
A framework for pretraining neural nets (alternative to random weight initialization).
I expect to implement Deep Belief Net pretraining using Stacked Restricted Boltzmann Machines (RBM)
and Stacked Sparse (Denoising) Autoencoders.
"""
abstract PretrainStrategy

# -----------------------------------------------------------------

immutable DenoisingAutoencoder <: PretrainStrategy end

# immutable DenoisingAutoencoder <: PretrainStrategy
#   net::NeuralNet
#   params::NetParams
#   solverParams::SolverParams
# end

# doc"""
# Using the NeuralNet layer and the parameters, construct a new NeuralNet which represents an autoencoder and
# which shares the weight/bias with the Layer
# """
# function makeAutoencoderNet(::Type{DenoisingAutoencoder}, layer::Layer, encoderParams::NetParams, solverParams::SolverParams)
#   outputlayer = Layer()
# end


function pretrain(::Type{DenoisingAutoencoder}, net::NeuralNet, sampler::DataSampler;
                    maxiter::Int = 1000,
                    dropout::DropoutStrategy = Dropout(pInput=0.7,pHidden=0.0),  # this is the "denoising" part, which throws out some of the inputs
                    encoderParams::NetParams = NetParams(η=0.1, μ=0.0, λ=0.0001, dropout=dropout),
                    solverParams::SolverParams = SolverParams(maxiter=maxiter, erroriter=typemax(Int), breakiter=typemax(Int)),  #probably don't set this manually??
                    inputActivation::Activation = IdentityActivation())

  # lets pre-load the input dataset for simplicity... just need the x vec, since we're trying to map: x --> somthing --> x
  dps = [DataPoint(dp.x, dp.x) for dp in DataPoints(sampler)]
  sampler = SimpleSampler(dps)

  # for each layer (which is not the output layer), fit the weights/bias as guided by the pretrain strategy
  for layer in net.layers[1:end-1]

    # build a neural net which maps: nin -> nout -> nin
    outputActivation = layer.activation
    autoencoder = buildNet(layer.nin, layer.nin, [layer.nout]; hiddenActivation=inputActivation, finalActivation=outputActivation, params=encoderParams)
    println("netlayer: $layer  oact: $outputActivation  autoenc: $autoencoder")

    # solve for the weights and bias... note we're not using stopping criteria... only maxiter
    stats = solve!(autoencoder, solverParams, sampler, sampler)
    println("  $stats")

    # save the weights and bias to the layer
    autoencoderlayer = autoencoder.layers[1]
    println("autoencoderlayer: $autoencoderlayer")
    layer.w = autoencoderlayer.w
    layer.b = autoencoderlayer.b

    # update the inputActivation, so that this layer's activation becomes the next autoencoder's inputActivation
    inputActivation = outputActivation

    # feed the data forward to the next layer
    for i in 1:length(dps)
      newx = forward(autoencoderlayer, dps[i].x, false)
      dps[i] = DataPoint(newx, newx)
    end

  end

  # we're done... net is pretrained now
  return
end


