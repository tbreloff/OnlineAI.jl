
# helper method so we don't calculate all positions for diagonal matrices
_cols_to_compute(A::Diagonal, i::Integer) = i:i
_cols_to_compute(A::AbstractMatrix, i::Integer) = 1:size(A,2)

# ----------------------------------------------------------------------------

function forward!(net::Circuit, x::AVec)

    # forward pass on input node
    copy!(net[1].s, x)
    forward!(net[1], compute_s = false)

    # forward pass for the rest (in order!)
    for i in 2:length(net)
        forward!(net[i])
    end

    # return the final output
    net[end].state.y
end

# TODO: the backward pass should compute ϕₒ = f'(sₒ) * dC/dyₒ  for final layer, then backprop the rest

# TODO: LearnBase should have a "sensitivity" function which does that ϕ calc... this way we can skip multiplying
#       deriv(mapping) * deriv(loss) for softmax/cross entropy

# backprop pass given error E(t)
function backward!(net::Circuit, input::AVec, target::AVec)

    # first compute the forward error responsibility of the final layer
    outputnode = net[end]
    output = outputnode.state.y
    ϕₒ = sensitivity!(outputnode.state.ϕ,
                      outputnode.mapping,
                      net.mloss,
                      input,
                      output,
                      target)
    backward!(outputnode, net.updater, net.ploss, compute_ϕ = false)

    # backprop
    for i in length(net)-1:-1:1
        backward!(net[i], net.updater, net.ploss)
    end

    # return the net
    net
end

# go forwards to compute outputs, then backwards to update Learnables
function OnlineStats.fit!(net::Circuit, input::AVec, target::AVec)
    ouput = forward!(net, input)
    backward!(net, input, target)
    net
end

# ----------------------------------------------------------------------------


function forward!{T}(node::Node{T}; compute_s::Bool = true)
    state = node.state

    # compute f_ratio(t-1) using previous state/output
    for i=1:node.n
        yi = state.y[i]
        state.f_ratio_prev[i] = yi == 0 ? 0 : deriv(node.mapping, state.s[i]) / yi
    end

    # when s_override is null, we must update the state.  otherwise don't update anything.
    # we will mostly use this for the input node
    if compute_s
        
        # first compute gates
        for gate in node.gates_in
            forward!(gate)
        end

        # compute the node state (sum of gates plus bias):
        #       sⱼ = ∑ sᵢ  +  bⱼ
        copy!(state.s, state.b)
        for gate in node.gates_in, i=1:node.n
            state.s[i] += gate.g[i]
        end

    end

    # apply the mapping function:
    #       yⱼ = fⱼ(sⱼ)
    value!(state.y, node.mapping, state.s)

    # return the output
    state.y
end

function backward!{T}(node::Node{T}, updater::ParameterUpdater, ploss::ParameterLoss; compute_ϕ::Bool = true)
    state = node.state

    # compute the sensitivity δⱼ = ∂C ./ ∂sⱼ
    #                            = (∂C ./ ∂sₒ) .* (∂sₒ ./ ∂sⱼ)
    #                            = δₒ .* ζⱼ
    
    if compute_ϕ

        # compute ϕ = error responsibility of feedforward connections
        # compute ψ = error responsibility of recurrent connections


    end

    # compute δ = ϕ + ψ
    for i in eachindex(state.δ)
        state.δ[i] = state.ϕ[i] + state.ψ[i]
    end

    # update the bias
    
end


# ----------------------------------------------------------------------------


function forward!(gate::Gate)
    state = gate.state

    # store ε(t-1) before we make any updates
    copy!(state.ε_prev, state.ε)
    
    # compute the eligibility trace for this gate:
    #       ε = x = ∏ yᵢ
    fill!(state.ε, 1)
    for node in gate.nodes_in, i=1:gate.n
        state.ε[i] *= node.state.y[i]
    end

    # next compute the state of the gate:
    #       s = w x = w ε
    state.g[:] = state.w * state.ε

    # return the output
    state.g
end

function backward!(gate::Gate, updater::ParameterUpdater, ploss::ParameterLoss, y::AVec, γ::AbstractFloat = 0.99)

    # don't do anything when the gatetype is FIXED
    if gate.gatetype == FIXED
        return gate
    end

    # note: deltahat from my notes is zeta: ζ
    # δₙ = δₒᵤₜ .* ζₙ
    state = gate.state
    n, m = size(state.w)

    # our goal is to calculate: ∇ᵢⱼ = γ ∇ᵢⱼ + δⱼ .* εᵢ
    # then we can update the weight matrix using the gradient updater
    for i=1:n, j in _cols_to_compute(state.w, i)
        state.∇[i,j] = γ * state.∇[i,j] + gate.node_out.state.δ[j] * state.ε[i]
        state.w[i,j] += param_change!(state.w_states[i,j], updater, ploss, state.∇[i,j], state.w[i,j])
    end

    gate
end

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
