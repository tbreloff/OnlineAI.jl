
function forward!(net::Circuit, x::AVec)
    # calculate activations, storing any state necessary for backward pass
end

function backward!(net::Circuit, y::AVec)
    # compute sensitivities and adjust weights/biases
end

function OnlineStats.fit!(net::Circuit, x::AVec, y::AVec)
    yhat = forward!(net, x)
    backward!(net, y)
end

