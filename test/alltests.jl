module test_OnlineAI

# reload("OnlineAI")
using OnlineAI
# include("../src/utils.jl")

function testxor(maxiter::Int)

  # all xor inputs and results
  inputs = float([0 0; 0 1; 1 0; 1 1])
  targets = Float64[(sum(row(inputs,i))==1.0)*0.8+0.1 for i in 1:size(inputs,1)]

  # all sets are the same
  data = buildSolverData(inputs, targets)
  datasets = DataSets(data, data, data)

  net = buildNeuralNet([2,2,1]; η=0.5, μ=0.1)

  params = buildSolverParams(maxiter=maxiter, minerror=1e-6)
  solve!(net, params, datasets)

  for d in data
    output = feedforward!(net, d.input)
    println("Result: input=$(d.input) target=$(d.target) output=$output")
  end

  net
end

# -------------------------------------------------------------------

using Qwt

sinfunction(x) = sin(x) / 2 + sin(x*4) / 4
# sinfunction(x) = 0.9 * x^2 / maxx^2
# sinfunction(x) = x^2

function updatesinplot(net::NeuralNet, params::SolverParams, datasets::DataSets, stats::SolverStats)

  setdata(plt.lines[1], xxx, Float64[predict(net, [f])[1] for f in xxx])
  setdata(plt.lines[2], xxx, map(sinfunction,xxx))
  title(plt, @sprintf("Update: %4d  Error: %10.4f", stats.numiter, stats.validationError))

  push!(plt2, 1, stats.numiter, stats.validationError)

  refresh(plt)
  refresh(plt2)
  
  shoulddogif && saveframe(anim)
end

function buildRegNN1()
  buildNeuralNet([1,10,10,1]; η=0.005, μ=0.1, activation=TanhMapping())
end

function buildRegNN2()
  η=0.02
  μ=0.2
  NeuralNet(Layer[
    buildLayer(1, 10),
    buildLayer(10, 10),
    buildLayer(10, 1, IdentityMapping())
    ], η, μ)
end


function testsin(maxiter::Int, dogif = false)

  global maxx = 10.0
  global sp = subplot(zeros(0,2), ncols=1)
  global plt = oplot(sp.plots[1], zeros(1,2), labels=["estimate","target"])
  global plt2 = oplot(sp.plots[2], zeros(1), xlabel="Iteration", ylabel="Network Error"); empty!(plt2)
  global xxx = collect(-maxx:.1:maxx)
  global anim = animation(sp, "/tmp/png")
  global shoulddogif = dogif

  inputs = collect(linspace(-maxx,maxx,1000))
  inputs = reshape(inputs, length(inputs), 1)
  targets = map(sinfunction, inputs)

  # all sets are the same
  data = buildSolverData(inputs, targets)
  datasets = DataSets(data, sample(data,30), data)

  # net = buildNeuralNet([1,10,10,1]; η=0.005, μ=0.1, activation=TanhMapping())
  net = buildRegressionNet(1, 1, [30,30]; hiddenMapping = ReLUMapping(), solver=NNetSolver(η = 0.05, dropout=DropoutStrategy(on=false)))

  params = SolverParams(maxiter=maxiter, onbreak=updatesinplot, displayiter=10000, minerror=1e-3)
  solve!(net, params, datasets)

  shoulddogif && makegif(anim)
  net
end


end
