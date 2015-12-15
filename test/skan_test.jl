
using OnlineAI
using Plots

F = Float32
S = SkanSynapse{F}
params = SkanParams{F}()

# init the neurons
numinputs = 2
inputs = [InputNeuron() for i in 1:numinputs]
numoutputs = 3
outputs = [SkanNeuron{F}(S[], S[], false, false, 0, 0, 40numinputs, 100numinputs) for i in 1:numoutputs]
neurons = join(inputs, outputs)

# init the synapses
synapses = [SkanSynapse{F}(input, output, 10000, 0, 0, 0, 0) for input in inputs, output in outputs]

# do a sim
n = 100
ramps = zeros(n, length(synapses))
potentials, thresholds = [zeros(n, numoutputs) for i=1:2]

for i in 1:100

    # create random spiking activity
    for input in inputs
        input.spike = rand() < 0.01
    end

    # update the network
    step!(params, neurons, synapses)

    for (j,s) in enumerate(synapses)
        ramps[i,j] = s.ramp
    end

    for (j,o) in enumerate(outputs)
        potentials[i,j] = o.potential
        thresholds[i,j] = o.threshold
    end

end


sp = subplot(plot(ramps), plot(potentials))
plot!(sp.plts[2], thresholds)



