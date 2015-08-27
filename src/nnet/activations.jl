
abstract Activation
forward(activation::Activation, Σs::AVecF) = Float64[forward(activation, Σ) for Σ in Σs]
backward(activation::Activation, Σs::AVecF) = Float64[backward(activation, Σ) for Σ in Σs]

function forward!(activation::Activation, a::AVecF, Σs::AVecF)
  for i in 1:length(a)
    a[i] = forward(activation, Σs[i])
  end
end
function backward!(activation::Activation, a::AVecF, Σs::AVecF)
  for i in 1:length(a)
    a[i] = backward(activation, Σs[i])
  end
end

# ---------------------------------------

doc"""
f(Σ) = Σ
f'(Σ) = 1
"""
immutable IdentityActivation <: Activation end
forward(activation::IdentityActivation, Σ::Float64) = Σ
backward(activation::IdentityActivation, Σ::Float64) = 1.0

doc"""
f(Σ) = 1 / (1 + exp(-Σ))
f'(Σ) = f(Σ) * (1 - f(Σ))
"""
immutable SigmoidActivation <: Activation end
forward(activation::SigmoidActivation, Σ::Float64) = 1.0 / (1.0 + exp(-Σ))
backward(activation::SigmoidActivation, Σ::Float64) = (s = forward(activation, Σ); s * (1.0 - s))

doc"""
f(Σ) = tanh(Σ)
f'(Σ) = 1 - tanh(Σ)²
"""
immutable TanhActivation <: Activation end
forward(activation::TanhActivation, Σ::Float64) = tanh(Σ)
backward(activation::TanhActivation, Σ::Float64) = 1.0 - tanh(Σ)^2

doc"""
f(Σ) = Σ / (1 + |Σ|)
f'(Σ) = 1 / (1 + |Σ|)²
"""
immutable SoftsignActivation <: Activation end
forward(activation::SoftsignActivation, Σ::Float64) = Σ / (1.0 + abs(Σ))
backward(activation::SoftsignActivation, Σ::Float64) = 1.0 / (1.0 + abs(Σ))^2

# Rectified Linear Unit
doc"""
f(Σ) = max(0, Σ)
f'(Σ) = { 1,  Σ > 0
        { 0,  Σ ≤ 0
"""
immutable ReLUActivation <: Activation end
forward(activation::ReLUActivation, Σ::Float64) = max(0.0, Σ)
backward(activation::ReLUActivation, Σ::Float64) = float(Σ > 0.0)

# Leaky Rectified Linear Unit: modified derivative to fix "dying ReLU" problem
doc"""
f(Σ) = max(0, Σ)
f'(Σ) = { 1,  Σ > 0
        { ρ,  Σ ≤ 0
"""
immutable LReLUActivation <: Activation
  ρ::Float64
end
LReLUActivation() = LReLUActivation(0.01)
forward(activation::LReLUActivation, Σ::Float64) = max(0.0, Σ)
backward(activation::LReLUActivation, Σ::Float64) = Σ > 0.0 ? 1.0 : activation.ρ

doc"""
UNTESTED
f(Σᵢ) = exp(Σᵢ) / Z
  where Z := sum(exp(Σ))
f'(Σᵢ) = f(Σᵢ) * (1 - f(Σᵢ))

Note: we expect the target vector to be a multinomal indicator vector, where 
a 1 in the iᵗʰ position implies that the instance belongs to the iᵗʰ class
"""
immutable SoftmaxActivation <: Activation end

function forward(activation::SoftmaxActivation, Σs::AVecF)
  evec = exp(Σs)
  evec / sum(evec)
end
function backward(activation::SoftmaxActivation, Σs::AVecF)
  error("This should never actually be used, as we expect Softmax to be used only with CrossEntropyCostModel and so we don't multiply by the derivative")
end


