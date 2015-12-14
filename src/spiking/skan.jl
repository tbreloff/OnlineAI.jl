
# Implementation of the SKAN neuron model from:
#   "Racing to Learn: Statistical Inference and Learning in a Single Spiking Neuron with Adaptive Kernels" Afshar et al 2014
#   "Turn Down That Noise: Synaptic Encoding of Afferent SNR in a Single Spiking Neuron" Afshar et al 2014


type SkanNeuron
    ramping::Int8   # p(t): state of the ramp-up/ramp-down flag {-1, 0, 1}
    spike::Bool     # s(t): the soma output (pulse)... binary: on or off
    θ::Int          # θ(t): threshold voltage

end


