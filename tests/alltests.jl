module test_OnlineAI

using OnlineAI
include("../src/utils.jl")

function testxor(maxiter::Int)

	# all xor inputs and results
	inputs = float([0 0; 0 1; 1 0; 1 1])
	targets = Float64[(sum(row(inputs,i))==1.0)*0.8+0.1 for i in 1:size(inputs,1)]

	# all sets are the same
	data = buildSolverData(inputs, targets)
	datasets = DataSets(data, data, data)

	nn = buildNeuralNet([2,2,1]; η=0.5, μ=0.1)

	params = buildSolverParams(maxiter=maxiter, minerror=1e-6)
	solve!(nn, params, datasets)

	for d in data
		output = feedforward!(nn, d.input)
		println("Result: input=$(d.input) target=$(d.target) output=$output")
	end

	nn
end

# -------------------------------------------------------------------

using Qwt
const maxx = 15.0
const sp = subplot(zeros(0,2), ncols=1)
const plt = oplot(sp.plots[1], zeros(1,2), labels=["estimate","target"])
const plt2 = oplot(sp.plots[2], zeros(1), xlabel="Iteration", ylabel="Network Error"); empty!(plt2)
const xxx = collect(-maxx:.1:maxx)
const anim = animation(sp, "/tmp/png")

# sinfunction(x) = sin(x) / 2 + sin(x*8) / 4
sinfunction(x) = 0.9 * x^2 / maxx^2

function updatesinplot(nn::NeuralNet, params::SolverParams, datasets::DataSets, stats::SolverStats)

	setdata(plt.lines[1], xxx, Float64[feedforward!(nn, [f])[1] for f in xxx])
	setdata(plt.lines[2], xxx, map(sinfunction,xxx))
	title(plt, @sprintf("Update: %4d  Error: %10.4f", stats.numiter, stats.validationError))

	push!(plt2, 1, stats.numiter, stats.validationError)

	refresh(plt)
	refresh(plt2)
	saveframe(anim)
end


function testsin(maxiter::Int)

	inputs = mat(collect(linspace(-maxx,maxx,1000)))
	targets = map(sinfunction, inputs)

	# all sets are the same
	data = buildSolverData(inputs, targets)
	datasets = DataSets(data, sample(data,30), data)

	nn = buildNeuralNet([1,10,10,1]; η=0.005, μ=0.1, activation=TanhActivation())

	params = buildSolverParams(maxiter=maxiter, onbreak=updatesinplot, displayiter=10000, minerror=0.01)
	solve!(nn, params, datasets)

	makegif(anim)
	nn
end


end
