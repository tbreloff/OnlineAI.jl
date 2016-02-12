
using OnlineAI
using MLPlots
plotly()
default(leg=false)

F = Float32
S = SkanSynapse{F}
params = SkanParams{F}(
    ramp_step_adjustment = 0,
    ramp_step_initfunc = () -> F(rand(1:2)),
    weight_init = 2
  )

txt = " ABC ACD ADA CDB"
letters = sort(unique(txt))
# letteridx = Dict([(l,i) for (i,l) in enumerate(letters)])
I = length(letters)

# init the neurons
numinputs = I
inputs = [InputNeuron() for i in 1:numinputs]
numoutputs = I
outputs = [SkanNeuron(S[],
                      false,
                      false,
                      F(0),
                      params.weight_init,
                      F(0),
                      F(0)
                      # F(20numinputs),
                      # F(100numinputs)
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
n = 100

synapse_fields = [:ramp,:ramp_step,:ramp_flag]
sdict = Dict(zip(synapse_fields, [zeros(n, length(synapses)) for i=1:length(synapse_fields)]))

ispikes = SpikeTrains(numinputs, c=:red, title="Input Spikes", ms=10)
ospikes = SpikeTrains(numoutputs, title="Output Spikes", ms=10)
opulses = SpikeTrains(numoutputs, title="Output Pulses", ms=10)

# neuron_fields = [:pulse, :spike, :potential, :threshold]
neuron_fields = [:potential, :threshold]
ndict = Dict(zip(neuron_fields, [zeros(n, numoutputs) for i=1:length(neuron_fields)]))
# opulses, spikes, potentials, thresholds = [zeros(n, numoutputs) for i=1:4]

# inputspikes = zeros(n, numinputs)



for t in 1:n

    # create random spiking activity
    # for input in inputs
    #     input.spike = rand() < 0.01
    # end

    # only the input neuron corresponding to the current letter spikes
    l = txt[mod1(round(Int,t), length(txt))]
    for (i,input) in enumerate(inputs)
      input.spike = l == letters[i]
    end


    # update the network
    step!(params, neurons, synapses)

    # store the network state for plotting
    for (j,s) in enumerate(synapses), (k,v) in sdict
        v[t,j] = getfield(s, k)
    end
    for (j,s) in enumerate(outputs), (k,v) in ndict
        v[t,j] = getfield(s, k)
    end
    for (j,s) in enumerate(outputs)
        s.pulse && push!(opulses, j, t)
        s.spike && push!(ospikes, j, t)
    end
    for (j,s) in enumerate(inputs)
        # inputspikes[t,j] = s.spike
        s.spike && push!(ispikes, j, t)
    end

end

splts = [plot(sdict[k], title=string(k)) for k in synapse_fields]
# nplts = [plot(ndict[k], title=k) for k in neuron_fields]
nplt = plot(ndict[:potential], title="Membrane Potential")
plot!(ndict[:threshold])
# inplt = plot(inputspikes,title="Inputs Spikes")

subplot(splts..., nplt, ispikes.plt, ospikes.plt, opulses.plt, size=(1000,1200), nc=1, link=true)



