# params should contain all algorithm-specific parameters and methods.
# at a minimum, we need to be able to compute the weight updates for a layer

abstract DropoutStrategy

immutable Dropout <: DropoutStrategy
  pInput::Float64  # the probability that a node is used for the weights from inputs
  pHidden::Float64  # the probability that a node is used for hidden layers
end
Dropout(; pInput=0.8, pHidden=0.5) = Dropout(pInput, pHidden)
Base.print(io::IO, d::Dropout) = print(io, "Dropout{$(d.pInput),$(d.pHidden)}")
Base.show(io::IO, d::Dropout) = print(io, d)
getDropoutProb(strat::Dropout, isinput::Bool) = isinput ? strat.pInput : strat.pHidden


immutable NoDropout <: DropoutStrategy end
Base.print(io::IO, d::NoDropout) = print(io, "NoDropout")
Base.show(io::IO, d::NoDropout) = print(io, d)
getDropoutProb(strat::NoDropout, isinput::Bool) = 1.0

# ----------------------------------------

type NetParams{GRAD<:ParameterUpdater, DROP<:DropoutStrategy, ERR<:Loss}
  updater::GRAD
  dropoutStrategy::DROP
  mloss::ERR
  weightInit::Function
end

function NetParams(;
                    updater::ParameterUpdater = AdaMaxUpdater(),
                    dropout::DropoutStrategy = NoDropout(),
                    mloss::Loss = L2DistLoss(),
                    weightInit::Function = _initialWeights
                  )
  NetParams(updater, dropout, mloss, weightInit)
end


Base.print(io::IO, p::NetParams) = print(io, "NetParams{$(p.updater) $(p.dropoutStrategy) $(p.mloss)}")
Base.show(io::IO, p::NetParams) = print(io, p)

# get the probability that we retain a node using the dropout strategy (returns 1.0 if off)
getDropoutProb(params::NetParams, isinput::Bool) = getDropoutProb(params.dropoutStrategy, isinput)
