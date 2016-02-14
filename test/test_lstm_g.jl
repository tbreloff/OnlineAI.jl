
include("../src/lstm_g/network.jl")
include("../src/lstm_g/viz.jl")

nin = 3
nhidden = 2
nout = 1

# basic ANN
inputlayer  = gatedlayer(3, tag=:input)
hiddenlayer = gatedlayer(2, tag=:hidden)
outputlayer = gatedlayer(1, tag=:output)
@show inputlayer hiddenlayer outputlayer

ci = connect(inputlayer, hiddenlayer)
co = connect(hiddenlayer, outputlayer)
@show ci co


# now add a gating gatedlayer
gatinglayer = gatedlayer(2, tag=:gater)
cg = connect(inputlayer, gatinglayer)

# gate ci
gate!(ci, gatinglayer)

# peephole uses SAME type... node i maps to node i
peep = connect(hiddenlayer, gatinglayer, SAME)

@show gatinglayer

net = GatedNetwork(inputlayer, hiddenlayer, GatedLayer{Float64}[inputlayer, gatinglayer, hiddenlayer, outputlayer])


