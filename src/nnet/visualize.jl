
type LayerViz
  layer::Layer
  circles::Vector{SceneItem}
  lines::Matrix{SceneItem}
end

# updates the color of the circles, and color/size of the weight connections
function OnlineStats.update!(layerviz::LayerViz, bigweight::Float64)
  # TODO: change the colors of the circles (activations) and the line widths (weights)
  # TODO: if we dropped out this node, color it black
  # TODO: add input values to a textbox hovering above the input lines
  # TODO: add Σ and activation values inside the circles of the nodes
end

# ----------------------

# getCenters(n, sz, radius) = getLinspace(n, sz / 2 - radius - 20)
getCenters(n, sz) = getLinspace(n+2, sz / 2)[2:end-1]

function visualize(net::NeuralNet)
  scene = Scene(P2(0,0), P2(800,600))
  L = length(net.layers)
  maxnodes = maximum(map(x->max(x.nin,x.nout), net.layers))
  W, H = size(scene)
  
  # figure out the biggest circle radius we can use
  maxHorizontalRadius = W / (L+2) / 3
  maxVerticalRadius = H / (maxnodes+1) / 3
  radius = min(maxHorizontalRadius, maxVerticalRadius)

  # get the x positions (first position is the inputs... won't need circles there)
  xs = getCenters(L+1, W)

  # create the layer visualizations
  defaultBrush!(:red)
  z = -500 # anything negative is fine... use this to put the lines behind the circles

  # add the nodes and weight connections
  vizs = LayerViz[]
  for (i, layer) in enumerate(net.layers)
    
    # get the xs/ys for the inputs and nodes for this layer
    xIn, xOut = xs[i:i+1]
    ysIn, ysOut = map(n->getCenters(n, H), (layer.nin, layer.nout))

    circles = [circle!(radius, P2(xOut, ysOut[iOut])) for iOut in 1:layer.nout]
    lines = [line!(circles[iOut], P3(xIn, ysIn[iIn], z)) for iOut in 1:layer.nout, iIn in 1:layer.nin]

    push!(vizs, LayerViz(layer, circles, lines))
  end

  # add the "input lines"
  nin = net.layers[1].nin
  ys = getCenters(nin, H)
  for i in 1:nin
    line!(P3(-W/2, ys[i], z), P3(xs[1], ys[i], z))
  end

  NetViz(net, vizs)
end


type NetViz
  net::NeuralNet
  layervizs::Vector{LayerViz}
end


function OnlineStats.update!(viz::NetViz)
  map(update!, viz.layervizs)
end
