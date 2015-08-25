- alternate solvers (solver.jl)
  - adagrad, nesterov?

- cross validation (data.jl)

- data samplers
  - maybe they can be OnlineStats, so that update! will add to the dataset
    - on this thought, we can stream data to a sampler which probablistically retains a limited subset of data points

- ensembles
  - should give a list of models, or parameter search.  update each model for a data point
  - on validation/scoring, we update the method of aggreation:
    - majority voting: need a way to throw out bad models
    - fit final model:  x = output of models (stacked horizontally), y = target
      - note: this could be a simple average, regression, or neural net

- hyperparameter search
  - grid search: fit a model for each gridpoint and return mapping (hyperparasms --> (model, stats))
  - fine-tuning: gradient descent on hyperparams?
  - GA, etc (initialize with coarse grid search? finalize with fine-tuning?)
