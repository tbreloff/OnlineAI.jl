
include("../src/lstm_g/layer.jl")

nin = 3
nhidden = 2
nout = 1

inputlayer = layer(3)
hiddenlayer = layer(2, tag=:hidden)
outputlayer = layer(1)
@show inputlayer hiddenlayer outputlayer

conn1 = connect(inputlayer, hiddenlayer)
conn2 = connect(hiddenlayer, outputlayer)
@show conn1 conn2

