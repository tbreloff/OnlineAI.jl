

abstract Node


# takes in nin inputs, outputs A(Σ), where Σ = dot(x,w) + b, and A is the activation
# NOTE: ignore bias! assume it's been passed in already
type Perceptron
	nin::Int
	A::Function
	dA::Function
	x::VecF
	w::VecF
	dw::VecF
	# δ::Float64
	Σ::Float64
end

function Perceptron(nin::Int, activationType::DataType, lossType::DataType)
	A, da = getActivationFunctions(activationType, lossType)
	Perceptron(nin, A, dA, zeros(nin), zeros(nin), zeros(nin), 0.0, 0.0)
end

function feedforward!(node::Perceptron, inputs::VecF)
	node.x = copy(inputs)
	node.Σ = dot(node.x, node.w)
	node.A(node.Σ)
end


# TODO: computeδ function(s)... can we have 1 generic function, or do we need output-specific one?

hiddenδ(node::Perceptron, weightedδ::Float64) = node.dA(node.Σ) * weightedδ
finalδ(node::Perceptron, err::Float64) = -node.dA(node.Σ) * err  # this probably needs to change for different loss functions?

function update!(node::Perceptron, δ::VecF, η::Float64, μ::Float64)
	dw = -η * node.x .* δ + μ * node.dw
	node.w += dw
	node.dw = dw
end

