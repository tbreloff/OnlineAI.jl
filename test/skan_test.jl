
using OnlineAI
using Plots; plotly()

F = Float32
S = SkanSynapse{F}
params = SkanParams{F}()

# init the neurons
numinputs = 2
inputs = [InputNeuron() for i in 1:numinputs]
numoutputs = 3
outputs = [SkanNeuron(S[], false, false, F(0), F(0), F(40numinputs), F(100numinputs)) for i in 1:numoutputs]
neurons = vcat(inputs, outputs)

# init the synapses
synapses = SkanSynapse[SkanSynapse(input, output, F(10000), zeros(F,3)..., zero(Int8)) for input in inputs, output in outputs]

# do a sim
n = 10000
ramps, ramp_steps = [zeros(n, length(synapses)) for i=1:2]
potentials, thresholds = [zeros(n, numoutputs) for i=1:2]

for i in 1:n

    # create random spiking activity
    for input in inputs
        input.spike = rand() < 0.01
    end

    # update the network
    step!(params, neurons, synapses)

    for (j,s) in enumerate(synapses)
        ramps[i,j] = s.ramp
        ramp_steps[i,j] = s.ramp_step
    end

    for (j,o) in enumerate(outputs)
        potentials[i,j] = o.potential
        thresholds[i,j] = o.threshold
    end

end

plts = map(plot, (ramps, ramp_steps, potentials, thresholds))
subplot(plts...)
# sp = subplot(plot(ramps), plot(potentials))
# plot!(sp.plts[2], thresholds)



