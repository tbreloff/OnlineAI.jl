



doc"""
A framework for pretraining neural nets (alternative to random weight initialization).
I expect to implement Deep Belief Net pretraining using Stacked Restricted Boltzmann Machines (RBM)
and Stacked Sparse (Denoising) Autoencoders.
"""
abstract PretrainStrategy


# default
function pretrain(net::NeuralNet, sampler::DataSampler, validationSampler::DataSampler; kwargs...)
  pretrain(DenoisingAutoencoder, net, sampler, validationSampler; kwargs...)
end


doc"Samples from the underlying sampler, but sets y = x.  Used in mapping inputs to themselves in autoencoders."
immutable AutoencoderDataSampler{T<:DataSampler} <: DataSampler
  sampler::T
end
StatsBase.sample(sampler::AutoencoderDataSampler) = (dp = sample(sampler.sampler); DataPoint(dp.x, dp.x))
DataPoints(sampler::AutoencoderDataSampler) = DataPoints([DataPoint(dp.x, dp.x) for dp in DataPoints(sampler.sampler)])

# -----------------------------------------------------------------

immutable DenoisingAutoencoder <: PretrainStrategy end



function pretrain(::Type{DenoisingAutoencoder}, net::NeuralNet, trainSampler::DataSampler, validationSampler::DataSampler;
                    tiedweights::Bool = true,
                    maxiter::Int = 10000,
                    dropout::DropoutStrategy = Dropout(pInput=0.7,pHidden=0.0),  # this is the "denoising" part, which throws out some of the inputs
                    encoderParams::NetParams = NetParams(dropout=dropout),
                    # solverParams::SolverParams = SolverParams(maxiter=maxiter, erroriter=typemax(Int), breakiter=typemax(Int)),  #probably don't set this manually??
                    solverParams::SolverParams = SolverParams(maxiter=maxiter, erroriter=2000, breakiter=2000, stopepochs=40),
                    inputActivation::Activation = IdentityActivation())

  # lets pre-load the input dataset for simplicity... just need the x vec, since we're trying to map: x --> somthing --> x
  # trainData = DataPoints([DataPoint(dp.x, dp.x) for dp in DataPoints(trainSampler)])
  # println(trainData)
  # trainSampler = SimpleSampler(trainData)
  
  # set up training data
  trainData = DataPoints(AutoencoderDataSampler(trainSampler))

  # set up validation data
  validationData = DataPoints(AutoencoderDataSampler(validationSampler))

  # for each layer (which is not the output layer), fit the weights/bias as guided by the pretrain strategy
  for layer in net.layers[1:end-1]

    # NOTE: we are mapping input --> hidden --> input, which is why the "center" activation is called hiddenActivation
    #       and the "final" activation is called inputActivation

    # some setup
    hiddenActivation = layer.activation
    inputTransformer = layer === first(net.layers) ? net.inputTransformer : IdentityTransformer()

    # build a neural net which maps: nin -> nout -> nin
    autoencoder = buildNet(layer.nin, layer.nin, [layer.nout];
                            hiddenActivation = hiddenActivation,
                            finalActivation = inputActivation,
                            params = encoderParams,
                            solverParams = solverParams,
                            inputTransformer = inputTransformer)

    # tied weights means w₂ = w₁' ... rebuild the layer with a TransposeView of the first layer's weights
    # l1, l2 = autoencoder.layers
    if tiedweights
      setSecondAutoencoderLayer(autoencoder, autoencoder.layers...)
      # gradientState = getGradientState(encoderParams.gradientModel, l2.nin, l2.nout)
      # autoencoder.layers[2] = Layer(l2.nin, l2.nout, l2.activation, l2.p,
      #                               l2.dwState,
      #                               l2.dbState,
      #                               l2.x, TransposeView(l1.w), l2.b, l2.δ,
      #                               l2.Σ, l2.a, l2.r, l2.nextr)
    end

    println("netlayer: $layer  oact: $inputActivation")
    println("autoenc: $autoencoder")

    # solve for the weights and bias... note we're not using stopping criteria... only maxiter
    stats = solve!(autoencoder, SimpleSampler(trainData), SimpleSampler(validationData), true)
    println("  $stats")

    if stats.bestModel == nothing || isnan(stats.bestValidationError) || isinf(stats.bestValidationError)
      warn("Somthing wrong with pretraining model: $stats")
    end

    # save the weights and bias to the layer
    # println("l1: $l1")

    l1 = first(stats.bestModel.layers)
    @assert isa(l1.w, Matrix)
    println(l1)
    layer.w = copy(l1.w)
    layer.b = copy(l1.b)

    # update the inputActivation, so that this layer's activation becomes the next autoencoder's inputActivation
    inputActivation = hiddenActivation

    # feed the data forward to the next layer
    for i in 1:length(trainData)
      x = transform(inputTransformer, trainData[i].x)
      forward!(l1, x, false)
      newx = copy(l1.a)
      trainData[i] = DataPoint(newx, newx)
    end

    # do the same for the validation set
    for i in 1:length(validationData)
      x = transform(inputTransformer, validationData[i].x)
      forward!(l1, x, false)
      newx = copy(l1.a)
      validationData[i] = DataPoint(newx, newx)
    end

    # println(trainData)

  end

  # we're done... net is pretrained now
  return
end


function setSecondAutoencoderLayer(autoencoder::NeuralNet, l1::Layer, l2::Layer)
  autoencoder.layers[2] = Layer(l2.nin, l2.nout, l2.activation, l2.p,
                                l2.dwState,
                                l2.dbState,
                                l2.x, TransposeView(l1.w), l2.b, l2.δ,
                                l2.Σ, l2.a, l2.r, l2.nextr)
end


function setSecondAutoencoderLayer(autoencoder::NeuralNet, l1::NormalizedLayer, l2::NormalizedLayer)
  autoencoder.layers[2] = NormalizedLayer(l2.nin, l2.nout, l2.activation, l2.p,
                                l2.dwState,
                                l2.dbState,
                                l2.dβState,
                                l2.dαState,
                                l2.xvar, l2.w,
                                l2.x, l2.xhat, l2.β, l2.α, l2.y, l2.δy,
                                l2.b, l2.Σ, l2.a, l2.δΣ,
                                l2.r, l2.nextr)
end






