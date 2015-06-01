

type NeuralNet
	layers::Vector{Layer}  # note: this doesn't include input layer!!
	η::Float64 # learning rate
	μ::Float64 # momentum
	activation::Activation
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

	NeuralNet(layers, η, μ, activation) 
end

function Base.show(io::IO, nn::NeuralNet)
	println(io, "NeuralNet{η=$(nn.η), μ=$(nn.μ), layers:")
	for layer in nn.layers
		println(io, "    ", layer)
	end
	println(io, "}")
end


# produces a vector of outputs (activations) from the network
function feedforward!(nn::NeuralNet, inputs::VecF)
	outputs = inputs
	for layer in nn.layers
		outputs = feedforward!(layer, outputs)
	end
	outputs
end


# given a vector of errors (true values - activations), update network weights
function backpropagate!(nn::NeuralNet, errors::VecF)

	# update δ (sensitivities)
	nextδ, nextw = finalδ!(nn.layers[end], errors)
	for i in length(nn.layers)-1:-1:1
		nextδ, nextw = hiddenδ!(nn.layers[i], nextδ, nextw)
	end

	# update weights
	for layer in nn.layers
		update!(layer, nn.η, nn.μ)
	end

end


function totalerror(nn::NeuralNet, inputs::VecF, targets::VecF)
	outputs = feedforward!(nn, inputs)
	0.5 * sumabs2(targets - outputs)
end


# online version
function update!(nn::NeuralNet, inputs::VecF, targets::VecF)
	outputs = feedforward!(nn, inputs)
	errors = targets - outputs
	backpropagate!(nn, errors)
	outputs
end


# batch version
function update!(nn::NeuralNet, inputs::MatF, targets::MatF)
	@assert size(inputs,2) == nn.nin
	@assert size(targets,2) == nn.nout
	@assert size(inputs,1) == size(targets,1)

	outputs = VecF[]
	for i in 1:size(inputs,1)
		output = update!(nn, row(inputs,i), row(targets,i))
		push!(outputs, output)
	end
	outputs
end

