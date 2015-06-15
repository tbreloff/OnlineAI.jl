

function findMatching(viznodes, neuron)
	for viznode in viznodes
		if viznode.neuron === neuron
			return viznode
		end
	end
	error("Couldn't find matching viznode for $neuron")
end

function getLinspace(n, h)
	h = n > 1 ? h : 0
	linspace(-h, h, n)
end


# ---------------------------------------------------------------------------

type LiquidVisualizationNode
	neuron::SpikingNeuron
	circle::SceneItem
end

function LiquidVisualizationNode(neuron::SpikingNeuron, pos::P3)
	# shift x/y coords based on zvalue to give a 3d-ish look
	z = pos[3]
	pos = pos + P3(z/5, z/5, 0)

	# create a new circle in the scene
	radius = 20
	circle = circle!(radius, pos)
	brush!(pen!(circle, 0), :lightGray)
	if !neuron.excitatory
		pen!(circle, 3, :yellow)
	end

	LiquidVisualizationNode(neuron, circle)
end

	# draw lines for synapses
function addSynapticConnections(viznodes::Vector{LiquidVisualizationNode})
	for viznode in viznodes
		for synapse in viznode.neuron.synapses
			connectedViznode = findMatching(viznodes, synapse.postsynapticNeuron)
			l = line!(viznode.circle, connectedViznode.circle)
			pen!(l, 1 + 2 * abs(synapse.weight), 0, 0, 0, 0.3)
		end
	end
end

# ---------------------------------------------------------------------------

function visualize(input::GRFInput, pos::P2, viznodes)
	ys = getLinspace(length(input.neurons), 100)
	pt = P3(pos - P2(70, 0), -10000)
	for (i,neuron) in enumerate(input.neurons)
		viznode = LiquidVisualizationNode(neuron, P3(pos + P2(0,ys[i])))
		push!(viznodes, viznode)
		pen!(line!(viznode.circle, pt), 2, :cyan)  # connect line to pt
	end
	pen!(line!(pt, pt - P3(70,0,0)), 2, :cyan)
end

# ---------------------------------------------------------------------------

type LiquidVisualization
	lsm::LiquidStateMachine
	window::Widget
	scene::Scene
	pltEstVsAct::PlotWidget
	pltScatter::PlotWidget
	viznodes::Vector{LiquidVisualizationNode}
	t::Int
end


function visualize(lsm::LiquidStateMachine)

	nin = lsm.nin
	nout = lsm.nout
	liquid = lsm.liquid

	# set up the liquid scene
	scene = Scene(show=false)
	background!(:gray)
	viznodes = LiquidVisualizationNode[]

	# input
	nin = length(lsm.inputs.inputs)
	x = -400
	ys = getLinspace(nin, 400)
	for (i,input) in enumerate(lsm.inputs.inputs)
		pos = P2(x, ys[i])
		visualize(input, pos, viznodes)
	end

	# liquid
	xs = getLinspace(liquid.params.l, 140)
	ys = getLinspace(liquid.params.w, 300)
	zs = getLinspace(liquid.params.h, 120)
	for neuron in liquid.neurons
		i, j, k = neuron.position
		pos = P3(xs[i], ys[j], zs[k])
		push!(viznodes, LiquidVisualizationNode(neuron, pos))
	end

	addSynapticConnections(viznodes)

	# set up the estimate vs actual plot
	pltEstVsAct = plot(zeros(0,nout*2),
										 title="predicted vs actual",
										 labels = map(i->string(i<=nout ? "Act" : "Est", i), 1:nout*2),
										 show=false)

	# set up the scatter plot of estimate vs actual
	pltScatter = scatter(zeros(0,nout),
											 zeros(0,nout),
											 xlabel = "predicted",
											 ylabel = "actual",
											 show=false)

	# put all 3 together into a widget container, resize, then show
	window = vsplitter(hsplitter(scene, pltScatter), pltEstVsAct)
	moveToLastScreen(window)
	resizewidget(window, screenSize(screenCount()) - P2(20,20))
	showwidget(window)

	LiquidVisualization(lsm, window, scene, pltEstVsAct, pltScatter, viznodes, 0)
end

#update visualization
function OnlineStats.update!(viz::LiquidVisualization, y::VecF)
	viz.t += 1

	for viznode in viz.viznodes
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

	est = predict(viz.lsm)
	for (i,e) in enumerate(est)
		push!(viz.pltEstVsAct, i, viz.t, y[i])
		push!(viz.pltEstVsAct, i+viz.lsm.nout, viz.t, e)
		push!(viz.pltScatter, i, e, y[i])
	end
	refresh(viz.pltEstVsAct)
	refresh(viz.pltScatter)

	sleep(0.0001)
end