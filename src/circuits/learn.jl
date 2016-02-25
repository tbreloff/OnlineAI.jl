
# helper method so we don't calculate all positions for diagonal matrices
_cols_to_compute(A::Diagonal, i::Integer) = i:i
_cols_to_compute(A::AbstractMatrix, i::Integer) = 1:size(A,2)

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
    state = node.state
    # TODO: special handling for input node... might want a special "input gate" for setting the xₜ

    # compute the node state (sum of gates plus bias):
    #       sⱼ = ∑ sᵢ  +  bⱼ
    copy!(state.s, state.b)
    for gate in node.gates_in, i=1:node.n
        state.s[i] += gate.s[i]
    end

    # and apply the activation function:
    #       yⱼ = fⱼ(sⱼ)
    forward!(node.activation, node.y, node.s)

    # return the node
    node
end

function backward!(node::Node, model::GradientModel)
    state = node.state
    # TODO: special handling for output node... might want a special "output gate" for δₒᵤₜ calc

    # compute the sensitivity δⱼ = ∂C ./ ∂sⱼ
    #                            = (∂C ./ ∂sₒᵤₜ) .* (∂sₒᵤₜ ./ ∂sⱼ)
    #                            = δₒᵤₜ .* ζⱼ

    
end


# ----------------------------------------------------------------------------


function forward!(gate::Gate)
    state = gate.state
    
    # first compute the eligibility trace for this gate:
    #       ε = ∏ yᵢ
    # then store the node output that generated the trace (for δ calc):
    #       yhatᵢ = yᵢ
    fill!(state.ε, 1)
    for node in gate.nodes_in, i=1:gate.n
        state.ε[i] *= node.state.y[i]
        state.yhat[i] = node.state.y[i]
    end

    # next compute the state of the gate:
    #       s = w * ε
    state.s[:] = state.w * state.ε

    # return the gate
    gate
end

function backward!(gate::Gate, y::AVec, model::GradientModel, γ::AbstractFloat = 0.99)

    # don't do anything when the gatetype is FIXED
    if gate.gatetype == FIXED
        return gate
    end

    # note: deltahat from my notes is zeta: ζ
    # δₙ = δₒᵤₜ .* ζₙ
    state = gate.state
    n, m = size(state.w)

    # our goal is to calculate: ∇ᵢⱼ = γ ∇ᵢⱼ + δⱼ .* εᵢ
    # then we can update the weight matrix using the gradient model
    for i=1:n, j in _cols_to_compute(state.w, i)
        state.∇[i,j] = γ * state.∇[i,j] + gate.node_out.state.δ[j] * state.ε[i]
        state.w[i,j] += Δij(model, state.gradient_state, state.∇[i,j], state.w[i,j], i, j)
    end

    gate
end

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
