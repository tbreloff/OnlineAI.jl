

using OnlineAI

function runtest(net, x, y, niter)
  for i in 1:niter
    fit!(net, x, y)
  end
end

nin, nout = 2, 1
hidden = [100,100]
params = NetParams(
    # gradientModel = SGDModel()
    gradientModel = AdaMaxModel()
  )
net = buildTanhClassificationNet(nin, nout, hidden; params = params)
x = randn(nin)
y = randn(nout)

runtest(net, x, y, 1)

Profile.clear()

# note: last pass this ran in ~8 secs
@time @profile runtest(net, x, y, 30000)
