

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
lookahead = 5
y = x[lookahead:end]
x = x[1:end-lookahead]
T = length(x)



# create liquid state machine
params = LiquidParams(λ = 1.5, w = 7, h = 3, decayRateDist = Uniform(0.99, 1.0), pctOutput = 0.7)
lsm = LiquidStateMachine(params, nin, nout)
liquid = lsm.liquid


# set up visualization
viz = visualize(lsm)


# fit the lsm
for t = 1:T
	update!(lsm, vec(y[t,:]), vec(x[t,:]))
	update!(viz, vec(y[t,:]))
end




# # plot the grf functions
# grfs = GaussianReceptiveField[input.grf for input in lsm.input.inputs[1,:]]
# plt3 = plot(grfs, -15.:.1:15.)


end

tl = testliquid
# lsm = tl.lsm
