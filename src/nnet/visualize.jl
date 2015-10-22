
const MAXNUMPLTS = 20

type NetProgressPlotter
  net::NeuralNet
  stats::SolverStats
  fields::Vector{Symbol}
  subplt::Plots.Subplot
end

# setup
function NetProgressPlotter(net::NeuralNet, stats::SolverStats, fields::Vector{Symbol} = Symbol[:x, :xhat, :y, :Σ, :a]; kw...)
  n = length(net.layers)
  m = length(fields)

  # initialize the plots
  plts = typeof(Plots.current())[]
  for field in fields
    for (i,layer) in enumerate(net.layers)
      arr = getfield(layer, field)
      push!(plts, scatter(min(length(arr), MAXNUMPLTS); title = "Layer $i: $field", leg=false, show=false, xticks=nothing, kw...))
    end
  end

  # create the subplot
  subplt = subplot(plts...; nc = length(net.layers), show=false, kw...)

  NetProgressPlotter(net, stats, fields, subplt)
end

function track_progress(net::NeuralNet; stats::SolverStats = SolverStats(), fields::AVec{Symbol} = Symbol[:x, :xhat, :y, :Σ, :a], kw...)
  NetProgressPlotter(net, stats, fields; kw...)
end

function OnlineStats.update!(plotter::NetProgressPlotter, updateiter = false; show=false)
  if updateiter
    plotter.stats.numiter += 1
  end

  for (i, field) in enumerate(plotter.fields)
    for (j, layer) in enumerate(plotter.net.layers)

      # get the values for each node in this layer
      arr = getfield(layer, field)
      
      # get the correct plot from the grid
      plt = plotter.subplt[i,j]

      # add the data
      data = vec(arr)[1:min(length(arr),MAXNUMPLTS)]
      push!(plt, plotter.stats.numiter, data)
    end
  end

  if show
    gui(plotter.subplt)
  end

  # todo: do something with other stats...
end


# # --------------------------------------------------------------------------


@require Qwt begin

  type LayerViz
    layer::NeuralNetLayer
    circles::Vector{Qwt.Ellipse}
    wgtlines::Matrix{Qwt.Line}
    biaslines::Vector{Qwt.Line}
    vals::Vector{Qwt.SceneText}
  end

  function updateLine(line::Qwt.Line, val::Float64, bigval::Float64)
    color = val > 0.0 ? :green : (val < 0.0 ? :red : :black)
    sz = max(1.0, 10.0 * abs(val) / bigval)
    Qwt.pen!(line, sz, color)
  end

  # updates the color of the circles, and color/size of the weight connections
  function OnlineStats.update!(viz::LayerViz, bigweight::Float64, bigbias::Float64)
    # change the colors of the line widths (weights)
    nout, nin = size(viz.wgtlines)
    l = viz.layer

    for i in 1:nout
      for j in 1:nin
        updateLine(viz.wgtlines[i,j], l.w[i,j], bigweight)
      end
      updateLine(viz.biaslines[i], l.b[i], bigbias)
    end

    # if we dropped out this node, color it black
    for i in 1:nout
      Qwt.brush!(viz.circles[i], l.nextr[i] == 0.0 ? :black : :white)
    end

    # add Σ and activation values inside the circles of the nodes
    for i in 1:nout
      s = @sprintf("Σ: %1.3f\na: %1.3f", l.Σ[i], forward(l.activation, l.Σ[i]))
      Qwt.settext(viz.vals[i], s)
      Qwt.position!(viz.vals[i], Qwt.position(viz.circles[i]))
    end

  end

  # ----------------------

  # getCenters(n, sz, radius) = getLinspace(n, sz / 2 - radius - 20)
  getCenters(n, sz) = getLinspace(n+2, sz / 2)[2:end-1]

  function visualize(net::NeuralNet)
    scene = Qwt.Scene(Qwt.P2(0,0), Qwt.P2(1200,800))
    L = length(net.layers)
    maxnodes = maximum(map(x->max(x.nin,x.nout), net.layers))
    W, H = size(scene)
    
    # figure out the biggest circle radius we can use
    maxHorizontalRadius = W / (L+2) / 2.5
    # maxVerticalRadius = H / (maxnodes+1) / 3
    # radius = min(maxHorizontalRadius, maxVerticalRadius)

    # get the x positions (first Qwt.position is the inputs... won't need circles there)
    xs = getCenters(L+1, W)

    # create the layer visualizations
    Qwt.defaultBrush!(:white)
    z = -500 # anything negative is fine... use this to put the wgtlines behind the circles

    # add the nodes and weight connections
    vizs = LayerViz[]
    for (i, layer) in enumerate(net.layers)
      
      # get the xs/ys for the inputs and nodes for this layer
      xIn, xOut = xs[i:i+1]
      ysIn, ysOut = map(n->getCenters(n, H), (layer.nin, layer.nout))

      radius = min(maxHorizontalRadius, H / (layer.nout+1) / 2.5)
      circles = [Qwt.circle!(radius, Qwt.P2(xOut, ysOut[iOut])) for iOut in 1:layer.nout]
      wgtlines = [Qwt.line!(circles[iOut], Qwt.P3(xIn, ysIn[iIn], z)) for iOut in 1:layer.nout, iIn in 1:layer.nin]

      biaspos = Qwt.P2((xIn + xOut)/2, -H/2)
      biaslines = [Qwt.line!(circles[iOut], biaspos) for iOut in 1:layer.nout]

      vals = [Qwt.text!("") for iOut in 1:layer.nout]

      push!(vizs, LayerViz(layer, circles, wgtlines, biaslines, vals))
    end

    # add the input textboxes
    nin = net.layers[1].nin
    ys = getCenters(nin, H)
    inputvalx = (-W/2 + xs[1]) / 2
    inputvals = [Qwt.text!("", Qwt.P2(inputvalx, ys[i])) for i in 1:nin]
    # for i in 1:nin
    #   Qwt.line!(Qwt.P3(-W/2, ys[i], z), Qwt.P3(xs[1], ys[i], z))
    # end

    viz = NetViz(net, vizs, inputvals)
    update!(viz)
    viz
  end


  type NetViz
    net::NeuralNet
    layervizs::Vector{LayerViz}
    inputvals::Vector
  end

  # find the biggest weight and bias in absolute value within the network... used for determining the line width
  biggestabswgt(net::NeuralNet) = maximum([maximum(abs(l.w)) for l in net.layers])
  biggestabsbias(net::NeuralNet) = maximum([maximum(abs(l.b)) for l in net.layers])

  function OnlineStats.update!(viz::NetViz; bigweight::Float64 = biggestabswgt(viz.net), bigbias::Float64 = biggestabsbias(viz.net))

    # update the layers
    for lv in viz.layervizs
      update!(lv, bigweight, bigbias)
    end

    # update the input val textboxes
    l = viz.net.layers[1]
    for i in 1:l.nin
      iv = viz.inputvals[i]
      Qwt.settext(iv, @sprintf("Input: %1.3f", l.x[i]))
    end

  end

end
