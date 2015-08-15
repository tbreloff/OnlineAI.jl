



# # takes in nin inputs, outputs A(Σ), where Σ = dot(x,w) + b, and A is the activation
# # NOTE: ignore bias! assume it's been passed in already
# type Perceptron <: Node
#   nin::Int
#   A::Function  # activation function
#   dA::Function # derivative
#   x::VecF  # nin x 1
#   w::VecF  # nin x 1
#   dw::VecF # nin x 1
#   δ::Float64
#   Σ::Float64
# end

# # TODO: change passing of activation
# # function Perceptron(nin::Int, activationType::DataType, lossType::DataType)
# function Perceptron(w::VecF, activation::Activation)
#   # A, da = getActivationFunctions(activationType, lossType)
#   nin = length(w)
#   Perceptron(nin, activation.A, activation.dA, zeros(nin), w, zeros(nin), 0.0, 0.0)
# end

# function feedforward!(node::Perceptron, inputs::VecF)
#   node.x = inputs
#   node.Σ = dot(node.x, node.w)
#   node.A(node.Σ)
# end


# # TODO: computeδ function(s)... can we have 1 generic function, or do we need output-specific one?

# function hiddenδ!(node::Perceptron, weightedδ::Float64)
#   node.δ = node.dA(node.Σ) * weightedδ
# end

# function finalδ!(node::Perceptron, err::Float64)
#   node.δ = -node.dA(node.Σ) * err  # this probably needs to change for different loss functions?
# end

# function OnlineStats.update!(node::Perceptron, η::Float64, μ::Float64)
#   dw = -η * node.δ * node.x + μ * node.dw
#   node.w += dw
#   node.dw = dw
# end

