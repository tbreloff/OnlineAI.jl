

type NeuralNet
	layers::Vector{Layer}  # note: this doesn't include input layer!!
	η::Float64 # learning rate
	μ::Float64 # momentum
end


# structure should include neuron counts for all layers, including input and output
function buildNeuralNet(structure::VecI;
												η::Float64 = 1e-2, 
												μ::Float64 = 0.0,
												activation::Activation = TanhActivation())
	@assert length(structure) > 1

	layers = Layer[]
	for i in 1:length(structure)-1
		nin, nout = structure[i:i+1]
		push!(layers, buildLayer(nin, nout, activation))
	end

	NeuralNet(layers, η, μ) 
end

function Base.show(io::IO, net::NeuralNet)
	println(io, "NeuralNet{η=$(net.η), μ=$(net.μ), layers:")
	for layer in net.layers
		println(io, "    ", layer)
	end
	println(io, "}")
end


# produces a vector of outputs (activations) from the network
function feedforward!(net::NeuralNet, inputs::VecF)
	outputs = inputs
	for layer in net.layers
		outputs = feedforward!(layer, outputs)
	end
	outputs
end


# given a vector of errors (true values - activations), update network weights
function backpropagate!(net::NeuralNet, errors::VecF)

	# update δ (sensitivities)
	finalδ!(net.layers[end], errors)
	for i in length(net.layers)-1:-1:1
		hiddenδ!(net.layers[i], net.layers[i+1])
	end

	# update weights
	for layer in net.layers
		update!(layer, net.η, net.μ)
	end

end


function totalerror(net::NeuralNet, inputs::VecF, targets::VecF)
	outputs = feedforward!(net, inputs)
	0.5 * sumabs2(targets - outputs)
end


# online version
function OnlineStats.update!(net::NeuralNet, inputs::VecF, targets::VecF)
	outputs = feedforward!(net, inputs)
	errors = targets - outputs
	backpropagate!(net, errors)
	outputs
end


# batch version
function OnlineStats.update!(net::NeuralNet, inputs::MatF, targets::MatF)
	@assert size(inputs,2) == net.nin
	@assert size(targets,2) == net.nout
	@assert size(inputs,1) == size(targets,1)

	outputs = VecF[]
	for i in 1:size(inputs,1)
		output = update!(net, row(inputs,i), row(targets,i))
		push!(outputs, output)
	end
	outputs
end

