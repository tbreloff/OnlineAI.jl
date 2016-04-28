
# abstract ParameterUpdater
# abstract ParameterUpdaterState

# # ----------------------------------------

"Allows for global storage of a ParameterUpdater, so that you don't need to pass it around."
type CurrentUpdater
  updater::ParameterUpdater
end

const _current_updater = CurrentUpdater(AdaMaxUpdater())

"Get the current global gradient updater used for gradient updates."
current_updater() = _current_updater.updater

"Set the current global gradient updater used for gradient updates."
current_updater!(updater::ParameterUpdater) = (_current_updater.updater = updater)


# # ----------------------------------------

"Allows for global storage of a ModelLoss, so that you don't need to pass it around."
type CurrentModelLoss
  mloss::ModelLoss
end

const _current_mloss = CurrentModelLoss(L2DistLoss())

"Get the current global gradient mloss used for gradient updates."
current_mloss() = _current_mloss.mloss

"Set the current global gradient mloss used for gradient updates."
current_mloss!(mloss::ModelLoss) = (_current_mloss.mloss = mloss)

# # ----------------------------------------

"Allows for global storage of a ParameterLoss, so that you don't need to pass it around."
type CurrentParameterLoss
  ploss::ParameterLoss
end

const _current_ploss = CurrentParameterLoss(NoParameterLoss())

"Get the current global gradient ploss used for gradient updates."
current_ploss() = _current_ploss.ploss

"Set the current global gradient ploss used for gradient updates."
current_ploss!(ploss::ParameterLoss) = (_current_ploss.ploss = ploss)



# --------------------------------------------------------------

"Enacts a strategy to adjust the learning rate"
abstract LearningRateModel

immutable FixedLearningRate <: LearningRateModel end
OnlineStats.fit!(lrmodel::FixedLearningRate, err::Float64) = nothing


"Adapts learning rate based on relative variance of the changes in the test error"
immutable AdaptiveLearningRate <: LearningRateModel
  updater::ParameterUpdater
  errordiff::Diff
  diffvar::Variance
  adjustmentPct::Float64
  cutoffRatio::Float64
end

function AdaptiveLearningRate(updater::ParameterUpdater,
                              adjustmentPct = 1e-2,
                              cutoffRatio = 1e-1;
                              wgt = ExponentialWeight(20))
  AdaptiveLearningRate(updater, Diff(), Variance(wgt), adjustmentPct, cutoffRatio)
end

# if the error is decreasing at a large rate relative to the variance, increase the learning rate (speed it up)
function OnlineStats.fit!(lrmodel::AdaptiveLearningRate, err::Float64)
  fit!(lrmodel.errordiff, err)
  fit!(lrmodel.diffvar, diff(lrmodel.errordiff))
  m = mean(lrmodel.diffvar)
  s = std(lrmodel.diffvar)
  if s > 0.0
    pct = lrmodel.adjustmentPct * (m / s < -lrmodel.cutoffRatio ? 1.0 : -1.0)
    lrmodel.updater.η *= (1.0 + pct)
  end
end
