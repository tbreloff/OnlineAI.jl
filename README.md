# OnlineAI.jl

Machine learning for sequential/streaming data

## This is a work in progress... use at your own risk!

Example usage:

```
# solve for the "xor" problem using a simple neural net with 1 hidden layer with 2 nodes
inputs = [0 0; 0 1; 1 0; 1 1]
targets = float(sum(inputs,2) .== 1)

# build train/validation/test sets, all with the same data
data = buildSolverData(float(inputs), targets)
datasets = DataSets(data, data, data)

# create the network with one 2-node hidden layer
# the solver defines some hyperparameters:
#   η := gradient descent speed
#   μ := momentum term
#   λ := L2-penalty param
hiddenLayerNodes = [2]
net = buildRegressionNet(ncols(inputs),
                         ncols(targets),
                         hiddenLayerNodes;
                         solver = NNetSolver(η=0.3, μ=0.1, λ=1e-5))

# some extra params for the solve iterations
params = SolverParams(maxiter=maxiter, minerror=1e-6)

# fit the net
solve!(net, params, datasets)

# now predict the output
output = predict(net, float(inputs))

# show it
for (o, d) in zip(output, data)
  println("Result: input=$(d.input) target=$(d.target) output=$o")
end
```

#### Implementation progress:

NNet:

- [x] Basic feedforward network
- [x] Backprop working
- [x] Standard activations/layers (Identity, Sigmoid, Tanh, Softsign)
- [ ] Other activations/layers (Softmax, ReLU, Dropout)
- [x] Basic data management (train/validate/test splitting)
- [ ] Advanced data cleaning/transformations (handling NaNs, map multinomal classes to dummies, standardizing)
- [x] Basic gradient descent solver (early stopping, momentum, L2 penalty)
- [x] Easy network building methods (buildClassifierNet, buildRegressionNet)
- [ ] Advanced network building methods (ReLU + dropout, multinomal classification)
- [ ] Generalized penalty functions
- [ ] Online algo: handle sequential data properly (unbiased validation/test data)
- [ ] Cross-validation framework
- [ ] Visualization tools (network design, connection weights, fit plots)

Experimental:

- [x] Spiking neuron model (Leaky Integrate and Fire Neuron based on Spike Response Model)
- [x] Gaussian receptive field for input spike train generation
- [x] Liquid State Machine (LSM) framework
- [x] LSM visualizations
- [ ] LSTM layer
- [ ] Grid-search for hyperparameters/net design
- [ ] GA for hyperparameters/net design
- [ ] Readout model tuning


## Roadmap/goals:
- Neural net framework with plug and play components
- Simple network building.  Give type of problem, desired input/output, and let it figure out a good network design
- Focus on time series and sequential models.
- Recurrent networks, time delay networks, LSTM.
- Mini-batch and single update solvers
- Spiking neural models
- Echo state networks / Liquid state machines
