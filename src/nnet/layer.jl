
# one layer of a multilayer perceptron (neural net)
# nodes == neurons,  ninᵢ == noutᵢ₋₁
type Layer
  nin::Int
  nout::Int
  nodes::Vector{Perceptron}
end

Base.print(io::IO, l::Layer) = print(io, "Layer{$(l.nin)=>$(l.nout)}")


initialWeights(nin::Int, nout::Int) = vcat((rand(nin) - 0.5) * 2.0 * sqrt(6.0 / (nin + nout)), 0.0)


function buildLayer(nin::Int, nout::Int, activation::Activation = SigmoidActivation())
  nodes = [Perceptron(initialWeights(nin, nout), activation) for j in 1:nout]
  Layer(nin, nout, nodes)
end



# takes input vector, and computes Σⱼ = wⱼ'x + bⱼ  and  Oⱼ = A(Σⱼ)
function feedforward!(layer::Layer, inputs::VecF)
  x = vcat(inputs, 1.0)  # add bias to the end
  Float64[feedforward!(node, x) for node in layer.nodes]
end

δ(layer::Layer, j::Int) = layer.nodes[j].δ
δ(layer::Layer) = Float64[δ(layer, j) for j in 1:layer.nout]

function hiddenδ!(layer::Layer, nextlayer::Layer)
  for j in 1:layer.nout
    weightedNextδ = dot(δ(nextlayer), Float64[node.w[j] for node in nextlayer.nodes])
    hiddenδ!(layer.nodes[j], weightedNextδ)
  end
end

function finalδ!(layer::Layer, errors::VecF)
  for j in 1:layer.nout
    finalδ!(layer.nodes[j], errors[j])
  end
end

function OnlineStats.update!(layer::Layer, η::Float64, μ::Float64)
  for node in layer.nodes
    update!(node, η, μ)
  end
end