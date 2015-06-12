

module testliquid

using OnlineStats, Qwt, Distributions, StatsBase, OnlineAI


# create data
nin = 1
nout = 1
T = 100
# x = collect(linspace(-15., 15., T))
x = randn(T)
for i in 2:T
	x[i] += x[i-1] * 0.5
end
y = sin(x)

plt1 = plot(x, ylabel="Inputs")
plt2 = plot(y, ylabel="Outputs", labels = map(x->"Output$x", 1:nout))
oplot(plt2, zeros(1,nout), labels = map(x->"Est$x", 1:nout))
foreach(plt2.lines[nout+1:end], empty!)


params = LiquidParams(λ = 1.5, w = 15, h = 3, decayRateDist = Uniform(0.97, 0.999))
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
viz = vsplitter(scene, hsplitter(plt1, plt2))
Qwt.moveWindowToCenterScreen(viz)
showwidget(viz)


startpos = P3(-300,-300,-300)
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
		push!(plt2, i+nout, t, e)
	end
	refresh(plt2)

	sleep(0.01)
end

grfs = GaussianReceptiveField[input.grf for input in lsm.input.inputs[1,:]]
plt3 = plot(grfs, -15.:.1:15.)

# type LiquidVisualization
# 	liquid::Liquid



# scene = currentScene()
# empty!(scene)
# background!(:gray)

# # draw axes.. by default it adds to current scene, but could pass scene as optional first arg
# line!(0, top(scene), 0, bottom(scene))
# line!(left(scene), 0, right(scene), 0)

# # cube of circles... connect all with lines
# startpos = P3(-300, -300, -300)
# endpos = P3(300, 300, 300)
# pdiff = endpos - startpos
# n = 3
# r = maximum(pdiff) / n / 5

# # create the circles
# circles = Array(Any, n, n, n)
# for i in 1:n
# 	for j in 1:n
# 		for k in 1:n
# 			# layout in a grid
# 			pos = startpos + (P3(i,j,k) - 1) .* pdiff ./ (n-1)

# 			# shift x/y coords based on zvalue to give a 3d-ish look
# 			z = pos[3]
# 			pos = pos + P3(z/8, z/6, 0)

# 			c = circles[i,j,k] = circle!(r, pos)
# 			brush!(pen!(c, 0), :lightGray)
# 		end
# 	end
# end

# # connect them randomly with lines (note the proper overlap based on z value)
# for c1 in circles
# 	for c2 in circles
# 		if rand() < 0.2
# 			l = line!(c1,c2)
# 			pen!(l, 2, 0, rand(), .8, .2)
# 		end
# 	end
# end

# # pretty lights :)
# for i in 1:500
# 	map(x->brush!(x, rand(3)...), circles)
# 	sleep(0.01)
# end


end

tl = testliquid
lsm = tl.lsm
