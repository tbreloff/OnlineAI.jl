
# ----------------------------------------------------------------------------

function forward!(net::Circuit, x::AVec)
    # calculate activations, storing any state necessary for backward pass
end

function backward!(net::Circuit, y::AVec)
    # compute sensitivities and adjust weights/biases
    # note: net should contain a GradientModel for updating 
end

function OnlineStats.fit!(net::Circuit, x::AVec, y::AVec)
    yhat = forward!(net, x)
    backward!(net, y)
end

# ----------------------------------------------------------------------------


function forward!(node::Node)
    # calculate activations, storing any state necessary for backward pass
    # TODO: special handling for input node... might want a special "input gate" for setting the xₜ
end

function backward!(node::Node, model::GradientModel)
    # compute sensitivities and adjust weights/biases
end


# ----------------------------------------------------------------------------


function forward!(gate::Gate)
    state = gate.state
    
    # first compute: ε = ∏ yᵢ
    fill!(state.ε, 1)
    for node in gate.nodes_in
        for i in 1:gate.n
            state.ε[i] *= node.state.y[i]
        end
    end

    # next compute: s = w * ε
    # note: we use gemv! to do:  s = 1 * w * ε + 0 * s
    BLAS.gemv!('N', 1.0, state.w, state.ε, 0.0, state.s)

    # return the gate
    gate
end

function backward!(gate::Gate, y::AVec, model::GradientModel, γ::AbstractFloat = 0.99)

    # don't do anything when the gatetype is FIXED
    if gate.gatetype == FIXED
        return gate
    end

    # deltahat from my notes is zeta: ζ
    # δₙ = δₒᵤₜ .* ζₙ
    state = gate.state
    n, m = size(state.w)

    # our goal is to calculate: ∇ᵢⱼ = γ ∇ᵢⱼ + δⱼ .* εᵢ
    # then we can update the weight matrix using the gradient model
    for i=1:n, j=1:m
        state.∇[i,j] = γ * state.∇[i,j] + gate.node_out.state.δ[j] * state.ε[i]
    end

    # now we can update the weight matrix
    for i=1:n, j=1:m
        dwij = Δij(model, state.gradient_state, state.)
    end

    gate
end

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
