

module testliquid

using OnlineStats, Qwt, Distributions, StatsBase, OnlineAI


# create data
nin = 1
nout = 1
T = 500
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

# plt1 = plot(x, ylabel="Inputs")
pltEstVsAct = plot(y, ylabel="Outputs", labels = map(x->"Output$x", 1:nout), show=false)
oplot(pltEstVsAct, zeros(0,nout), labels = map(x->"Est$x", 1:nout))
# foreach(pltEstVsAct.lines[nout+1:end], empty!)

pltScatter = scatter(zeros(0,nout), zeros(0,nout), show=false)


params = LiquidParams(λ = 1.5, w = 7, h = 3, decayRateDist = Uniform(0.995, 1.0), pctOutput = 1.0)
lsm = LiquidStateMachine(params, nin, nout)
liquid = lsm.liquid


type LiquidVisualizationNode
	neuron
	circle
end


scene = currentScene()
empty!(scene)
background!(:gray)


# put all 3 together
# viz = vsplitter(scene, hsplitter(plt1, pltEstVsAct))
viz = vsplitter(hsplitter(scene, pltScatter), pltEstVsAct)
Qwt.moveWindowToCenterScreen(viz)
resizewidget(viz, P2(2000,1500))
showwidget(viz)


startpos = P3(-200,-200,-300)
diffpos = -2 * startpos
radius = 20

convertIdxToPct(i, n) = (i-1) / (n-1)

# set up the nodes of the visualization
liquidsz = (liquid.params.l, liquid.params.w, liquid.params.h)
viznodes = Array(LiquidVisualizationNode, liquidsz...)
for neuron in liquid.neurons
	pct = (P3(neuron.position...) - 1) ./ (P3(liquidsz...) - 1)
	pos = startpos + pct .* diffpos

	# shift x/y coords based on zvalue to give a 3d-ish look
	z = pos[3]
	pos = pos + P3(z/6, z/3, 0)

	# add the circles
	circle = circle!(20, pos)
	brush!(pen!(circle, 0), :lightGray)
	if !neuron.excitatory
		pen!(circle, 3, :yellow)
	end

	viznodes[neuron.position...] = LiquidVisualizationNode(neuron, circle)
end

# draw lines for synapses
for viznode in viznodes
	for synapse in viznode.neuron.synapses
		l = line!(viznode.circle, viznodes[synapse.postsynapticNeuron.position...].circle)
		pen!(l, 1 + 2 * abs(synapse.weight), 0, 0, 0, 0.3)
	end
end


# fit the lsm
for t = 1:T
	update!(lsm, vec(y[t,:]), vec(x[t,:]))

	#update visualization
	for viznode in viznodes
		neuron = viznode.neuron
		local args
		if neuron.fired
			args = (:red,)
		else
			upct = 1 - neuron.u / neuron.ϑ
			args = (upct,upct,upct)
		end
		brush!(viznode.circle, args...)
	end

	est = predict(lsm)
	for (i,e) in enumerate(est)
		push!(pltEstVsAct, i+nout, t, e)
		push!(pltScatter, i, e, y[t,i])
	end
	refresh(pltEstVsAct)
	refresh(pltScatter)

	sleep(0.0001)
end

# # plot the grf functions
# grfs = GaussianReceptiveField[input.grf for input in lsm.input.inputs[1,:]]
# plt3 = plot(grfs, -15.:.1:15.)


end

tl = testliquid
lsm = tl.lsm
