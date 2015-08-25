

abstract CostModel

function cost(model::CostModel, y::AVecF, yhat::AVecF)
  sum([cost(model, y[i], yhat[i]) for i in 1:length(y)])
end

# note: the vector version of costMultiplier also returns a boolean which is true when we
#       need to multiply this value by f'(Σ) when calculating the sensitivities δ
function costMultiplier(model::CostModel, y::AVecF, yhat::AVecF)
  Float64[costMultiplier(model, y[i], yhat[i]) for i in length(y)], true
end


#-------------

doc"""
Typical sum of squared errors:
  cost = sumabs2(yhat-y)
"""
immutable L2CostModel <: CostModel end

"cost function"
cost(::L2CostModel, y::Float64, yhat::Float64) = 0.5 * (y - yhat) ^ 2
cost(::L2CostModel, y::AVecF, yhat::AVecF) = 0.5 * sumabs2(y - yhat)

"returns M from the equation δ = M .* f'(Σ) ... used in the gradient update"
costMultiplier(::L2CostModel, y::Float64, yhat::Float64) = yhat - y

#-------------

"cost = sumabs(yhat-y)"
immutable L1CostModel <: CostModel end

cost(::L1CostModel, y::Float64, yhat::Float64) = abs(y - yhat)
costMultiplier(::L1CostModel, y::Float64, yhat::Float64) = sign(yhat - y)

#-------------

"""
Typical sum of squared errors, but scaled by ρ*y.  We implicitly assume that y ∈ {0,1}.
"""
immutable WeightedL2CostModel <: CostModel
  ρ::Float64
end
cost(model::WeightedL2CostModel, y::Float64, yhat::Float64) = 0.5 * (y - yhat) ^ 2 * (y > 0 ? model.ρ : 1)
costMultiplier(model::WeightedL2CostModel, y::Float64, yhat::Float64) = (yhat - y) * (y > 0 ? model.ρ : 1)

#-------------

immutable CrossEntropyCostModel <: CostModel end

cost(model::CrossEntropyCostModel, y::Float64, yhat::Float64) = -log(y > 0.0 ? yhat : (1.0 - yhat)) # binary case
function cost(model::CrossEntropyCostModel, y::AVecF, yhat::AVecF) # softmax case
  length(y) == 1 && return cost(model, y[1], yhat[1])
  C = 0.0
  for (i,yi) in enumerate(y)
    C -= yi * log(yhat[i])
  end
  C
end

costMultiplier(model::CrossEntropyCostModel, y::Float64, yhat::Float64) = yhat - y # binary case
function costMultiplier(model::CrossEntropyCostModel, y::AVecF, yhat::AVecF) # softmax case
  (length(y) == 1 ? Float64[costMultiplier(model, y[1], yhat[1])] : yhat - y), false
end


#-------------
