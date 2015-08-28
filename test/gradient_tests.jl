
using OnlineAI, FactCheck
import OnlineStats: @LOG

A = SigmoidActivation()
# A = TanhActivation()
C = CrossEntropyCostModel()
# C = L2CostModel()

nin, nout = 3, 1
hidden = [2]
net = buildClassificationNet(nin, nout, hidden;
                              finalActivation = A,
                              params = NetParams(costModel = C)
                            )

x = rand(nin) - 0.5
y = [1.0]

ε = 1e-4
numgrads = []
ls = net.layers

for l in ls
  w = l.w
  numgrad = similar(w)  # numerical gradient := (C(wij + ε) - C(wij - ε)) / (2ε)
  for i in 1:nrows(w)
    for j in 1:ncols(w)
      wij = w[i,j]

      w[i,j] = wij + ε
      cp = cost(net, x, y)

      w[i,j] = wij - ε
      cm = cost(net, x, y)

      numgrad[i,j] = (cp - cm) / (2ε)

      # put wij back
      w[i,j] = wij
    end
  end
  push!(numgrads, numgrad)
end


yhat = forward!(net, x, true)
multiplyDerivative = OnlineAI.costMultiplier!(net.params.costModel, net.costmult, y, yhat)
OnlineAI.updateSensitivities!(ls[end], net.costmult, multiplyDerivative)
for i in length(ls)-1:-1:1
  OnlineAI.updateSensitivities!(ls[i:i+1]...)
end

δs = [l.δ for l in ls]
xs = [l.x for l in ls]
Σs = [l.Σ for l in ls]
ws = [l.w for l in ls]
bs = [l.b for l in ls]

@show x
@show y
@show yhat
@show net.costmult
@show multiplyDerivative

facts("gradient") do
  for i in 1:length(ls)
    println()
    @show i
    @show xs[i]
    @show δs[i]
    @show ws[i]
    @show bs[i]
    @show Σs[i]
    @show numgrads[i]
    @show δs[i] * xs[i]'

    @fact δs[i] * xs[i]' --> roughly(numgrads[i], rtol=1e-3)
  end
end


# now update the weights
for l in ls
  OnlineAI.updateWeights!(l, net.params.gradientModel)
end


ws = [l.w for l in ls]
bs = [l.b for l in ls]

for i in 1:length(ls)
  println()
  @show i
  @show ws[i]
  @show bs[i]
end
