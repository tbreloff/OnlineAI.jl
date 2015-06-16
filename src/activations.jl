
immutable Activation
  A::Function  # activation function
  dA::Function # derivative
end

returnone(x) = 1.0
IdentityActivation() = Activation(nop, returnone)

sigmoid(y::Float64) = 1.0 / (1.0 + exp(-y))
sigmoidprime(y::Float64) = (sy = sigmoid(y); sy * (1.0 - sy))
SigmoidActivation() = Activation(sigmoid, sigmoidprime)

tanhprime(y::Float64) = 1.0 - tanh(y)^2
TanhActivation() = Activation(tanh, tanhprime)

softsign(y::Float64) = y / (1 + abs(y))
softsignprime(y::Float64) = 1 / (1 + abs(y))^2
SoftsignActivation() = Activation(softsign, softsignprime)

