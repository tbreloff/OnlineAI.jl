
abstract Activation
forward(activation::Activation, Σs::AVecF) = Float64[forward(activation, Σ) for Σ in Σs]
backward(activation::Activation, Σs::AVecF) = Float64[backward(activation, Σ) for Σ in Σs]


immutable IdentityActivation <: Activation end
forward(activation::IdentityActivation, Σ::Float64) = Σ
backward(activation::IdentityActivation, Σ::Float64) = 1.0

immutable SigmoidActivation <: Activation end
forward(activation::SigmoidActivation, Σ::Float64) = 1.0 / (1.0 + exp(-Σ))
backward(activation::SigmidActivation, Σ::Float64) = (s = forward(activation, Σ); s * (1.0 - s))

immutable TanhActivation <: Activation end
forward(activation::TanhActivation, Σ::Float64) = tanh(Σ)
backward(activation::TanhActivation, Σ::Float64) = 1.0 - tanh(Σ)^2

immutable SoftsignActivation <: Activation end
forward(activation::SoftsignActivation, Σ::Float64) = Σ / (1.0 + abs(Σ))
backward(activation::SoftsignActivation, Σ::Float64) = 1.0 / (1.0 + abs(Σ))^2



# immutable Activation
#   A::Function  # activation function
#   dA::Function # derivative
# end

# returnone(x) = 1.0
# IdentityActivation() = Activation(nop, returnone)

# sigmoid(y::Float64) = 1.0 / (1.0 + exp(-y))
# sigmoidprime(y::Float64) = (sy = sigmoid(y); sy * (1.0 - sy))
# SigmoidActivation() = Activation(sigmoid, sigmoidprime)

# tanhprime(y::Float64) = 1.0 - tanh(y)^2
# TanhActivation() = Activation(tanh, tanhprime)

# softsign(y::Float64) = y / (1 + abs(y))
# softsignprime(y::Float64) = 1 / (1 + abs(y))^2
# SoftsignActivation() = Activation(softsign, softsignprime)

