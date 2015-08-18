
type LayerViz
  layer::Layer
  circles::Vector{SceneItem}
  lines::Matrix{SceneItem}
end

function OnlineStats.update!(layerviz::LayerViz)
  # TODO: change the colors of the circles (activations) and the line widths (weights)
end

# ----------------------

getCenters(n, sz, radius) = getLinspace(n, sz / 2 - radius - 20)

function visualize(net::NeuralNet)
  scene = Scene(P2(0,0), P2(800,600))
  L = length(net.layers)
  maxnodes = maximum(map(x->max(x.nin,x.nout), net.layers))
  W, H = size(scene)
  
  # figure out the biggest circle radius we can use
  maxHorizontalRadius = W / (L+1) / 2
  maxVerticalRadius = H / maxnodes / 2
  radius = min(maxHorizontalRadius, maxVerticalRadius)

  # get the x positions (first position is the inputs... won't need circles there)
  # xs = getLinspace(L+1, W / 2 - radius - 20)
  xs = getCenters(L+1, W, radius)

  # create the layer visualizations
  vizs = [LayerViz(net.layer[i], SceneItem[], SceneItem[]) for i in 1:L]

  # add the nodes and weight connections
  for (i, viz) in enumerate(vizs)
    l = viz.layer
    
    # get the xs/ys for the inputs and nodes for this layer
    xIn, xOut = xs[i:i+1]
    ysIn, ysOut = map(x->getCenters(x, H, radius), (l.nin, l.nout))

    viz.circles = [circle!(radius, P2(xOut, ysOut[iOut])) for iOut in 1:l.nout]
    viz.lines = [line!(viz.circles[iOut], P2(xIn, ysIn[iIn])) for iOut in 1:l.nout, iIn in 1:l.nin]
  end

end


type NetViz
  net::NeuralNet
  layervizs::Vector{LayerViz}
end