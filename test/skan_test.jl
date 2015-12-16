
using OnlineAI
using Plots
# plotly()
default(leg=false)

F = Float32
S = SkanSynapse{F}
params = SkanParams{F}()

# init the neurons
numinputs = 2
inputs = [InputNeuron() for i in 1:numinputs]
numoutputs = 1
outputs = [SkanNeuron(S[],
                      false,
                      false,
                      F(0),
                      params.weight_init,
                      F(40numinputs),
                      F(100numinputs)
                     ) for i in 1:numoutputs]
neurons = vcat(inputs, outputs)

# init the synapses
synapses = SkanSynapse[]
for input in inputs, output in outputs
    synapse = SkanSynapse(input,
                          output,
                          params.weight_init,
                          zero(F),
                          zero(F),
                          params.ramp_step_initfunc(),
                          zero(Int8))
    push!(output.incoming_synapses, synapse)
    push!(synapses, synapse)
end


# do a sim
n = 10000

synapse_fields = [:ramp,:ramp_step,:ramp_flag]
sdict = Dict(zip(synapse_fields, [zeros(n, length(synapses)) for i=1:length(synapse_fields)]))

neuron_fields = [:pulse, :spike, :potential, :threshold]
ndict = Dict(zip(neuron_fields, [zeros(n, numoutputs) for i=1:length(neuron_fields)]))
# pulses, spikes, potentials, thresholds = [zeros(n, numoutputs) for i=1:4]

inputspikes = zeros(n, numinputs)

for i in 1:n

    # create random spiking activity
    for input in inputs
        input.spike = rand() < 0.01
    end

    # update the network
    step!(params, neurons, synapses)

    # store the network state for plotting
    for (j,s) in enumerate(synapses), (k,v) in sdict
        v[i,j] = getfield(s, k)
    end
    for (j,s) in enumerate(outputs), (k,v) in ndict
        v[i,j] = getfield(s, k)
    end
    for (j,s) in enumerate(inputs)
        inputspikes[i,j] = s.spike
    end

end

splts = [plot(sdict[k], title=k) for k in synapse_fields]
nplts = [plot(ndict[k], title=k) for k in neuron_fields]
inplt = plot(inputspikes,title="Inputs Spikes")

subplot(splts..., nplts..., inplt, size=(1000,1200), nc=1, link=true)



