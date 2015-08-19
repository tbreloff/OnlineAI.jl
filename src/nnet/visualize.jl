
type LayerViz
  layer::Layer
  circles::Vector{Ellipse}
  wgtlines::Matrix{Line}
  biaslines::Vector{Line}
  vals::Vector{SceneText}
end

function updateLine(line::Line, val::Float64, bigval::Float64)
  color = val > 0.0 ? :green : (val < 0.0 ? :red : :black)
  sz = max(1.0, 10.0 * abs(val) / bigval)
  pen!(line, sz, color)
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
    brush!(viz.circles[i], l.nextr[i] == 0.0 ? :black : :white)
  end

  # add Σ and activation values inside the circles of the nodes
  for i in 1:nout
    s = @sprintf("Σ: %1.3f\na: %1.3f", l.Σ[i], forward(l.activation, l.Σ[i]))
    settext(viz.vals[i], s)
    position!(viz.vals[i], position(viz.circles[i]))
  end

end

# ----------------------

# getCenters(n, sz, radius) = getLinspace(n, sz / 2 - radius - 20)
getCenters(n, sz) = getLinspace(n+2, sz / 2)[2:end-1]

function visualize(net::NeuralNet)
  scene = Scene(P2(0,0), P2(1200,800))
  L = length(net.layers)
  maxnodes = maximum(map(x->max(x.nin,x.nout), net.layers))
  W, H = size(scene)
  
  # figure out the biggest circle radius we can use
  maxHorizontalRadius = W / (L+2) / 2.5
  # maxVerticalRadius = H / (maxnodes+1) / 3
  # radius = min(maxHorizontalRadius, maxVerticalRadius)

  # get the x positions (first position is the inputs... won't need circles there)
  xs = getCenters(L+1, W)

  # create the layer visualizations
  defaultBrush!(:white)
  z = -500 # anything negative is fine... use this to put the wgtlines behind the circles

  # add the nodes and weight connections
  vizs = LayerViz[]
  for (i, layer) in enumerate(net.layers)
    
    # get the xs/ys for the inputs and nodes for this layer
    xIn, xOut = xs[i:i+1]
    ysIn, ysOut = map(n->getCenters(n, H), (layer.nin, layer.nout))

    radius = min(maxHorizontalRadius, H / (layer.nout+1) / 2.5)
    circles = [circle!(radius, P2(xOut, ysOut[iOut])) for iOut in 1:layer.nout]
    wgtlines = [line!(circles[iOut], P3(xIn, ysIn[iIn], z)) for iOut in 1:layer.nout, iIn in 1:layer.nin]

    biaspos = P2((xIn + xOut)/2, -H/2)
    biaslines = [line!(circles[iOut], biaspos) for iOut in 1:layer.nout]

    vals = [text!("") for iOut in 1:layer.nout]

    push!(vizs, LayerViz(layer, circles, wgtlines, biaslines, vals))
  end

  # add the input textboxes
  nin = net.layers[1].nin
  ys = getCenters(nin, H)
  inputvalx = (-W/2 + xs[1]) / 2
  inputvals = [text!("", P2(inputvalx, ys[i])) for i in 1:nin]
  # for i in 1:nin
  #   line!(P3(-W/2, ys[i], z), P3(xs[1], ys[i], z))
  # end

  viz = NetViz(net, vizs, inputvals)
  update!(viz)
  viz
end


type NetViz
  net::NeuralNet
  layervizs::Vector{LayerViz}
  inputvals::Vector{}
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
    settext(iv, @sprintf("Input: %1.3f", l.x[i]))
  end

end
