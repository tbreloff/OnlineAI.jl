
u = 0
spikes = [1, 4, 8, 15]
tstep = 0.1

α(q, s) = q / (τs - τr) * exp(-s / τs)

current = 0.
for t in tstep:tstep:30.
  du = 
  ut+1 = (1-lambda) * u
  du = lambda * u
end




# -----------------------------------------------------------------------
# general algo for each timestep:

foreach neuron
  step!(neuron)  # this should decay u towards urest, as well as setting n.fired=false... 
                  # also decay q (:= total current (pulse)) from all synapses towards 0, then add it to u
end

while true
  didfire = false
  foreach neuron
    if u >= threshold
      
      # reset neuron
      didfire = true
      fired=true
      u = u_refractory

      # transmit pulse
      foreach n.synapses
        pn = synapse.postneuron
        pulse = basepulse * syn.weight 
        pn.q += pulse

        # don't increate u if it already fired
        if !pn.fired
          pn.u += pulse
        end
      end
    end
  end
end

# now that we broke out, everything has stepped forward, fired, and transmitted

