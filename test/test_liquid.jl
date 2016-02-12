

module testliquid

using OnlineStats, Qwt, Distributions, StatsBase, OnlineAI


# create data

nin = 1
nout = 1
T = 2000
# x = collect(linspace(-15., 15., T))

# x is a simple differenced AR(1)... y is the future val
x = randn(T)
for i in 2:T
  x[i] += x[i-1] * 0.95
end
# dx = diff(x)
lookahead = 100
y = x
# y = x[lookahead:end]
# x = x[1:end-lookahead]
T = length(x)



# create liquid state machine
params = LiquidParams(λ = 1.0, 
                      w = 8,
                      h = 4,
                      decayRateDist = Uniform(0.99, 1.0),
                      pctInput = 0.2,
                      pctOutput = 0.8,
                      readout = FireWindowReadout(30),
                      baseThreshold = 1.0)
                      # readout = StateReadout())
lsm = LiquidStateMachine(params, nin, nout)
# liquid = lsm.liquid


# set up visualization
viz = visualize(lsm)


# fit the lsm
plotiter = 10
# anim = animation(viz.window, "/Users/tom/Pictures/gifs")
for (i,t) = enumerate(lookahead:T)
  fit!(lsm, vec(x[t-lookahead+1,:]), vec(y[t,:]))
  
  if i % plotiter == 0
    fit!(viz, vec(y[t,:]))
    # saveframe(anim)
  end
end
# makegif(anim)




# # plot the grf functions
# grfs = GaussianReceptiveField[input.grf for input in lsm.input.inputs[1,:]]
# plt3 = plot(grfs, -15.:.1:15.)


end

tl = testliquid
# lsm = tl.lsm
