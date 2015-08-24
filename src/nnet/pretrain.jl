
doc"""
A framework for pretraining neural nets (alternative to random weight initialization).
I expect to implement Deep Belief Net pretraining using Stacked Restricted Boltzmann Machines (RBM)
and Stacked Sparse (Denoising) Autoencoders.
"""
function pretrain(net::NeuralNet, strat::PretrainStrategy, sampler::DataSampler)
  # for each layer (which is not the output layer), fit the weights/bias as guided by the pretrain strategy


  for 
end



abstract PretrainStrategy

immutable DenoisingAutoencoderStrategy <: PretrainStrategy
end



