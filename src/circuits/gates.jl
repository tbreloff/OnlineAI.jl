

_initial_weights{T <: AbstractFloat}(::Type{T}, nin::Integer, nout::Integer, gatetype::Val{ALL}) = map(T, 0.5randn(nout, nin) / sqrt(nin))
_initial_weights{T <: AbstractFloat}(::Type{T}, nin::Integer, nout::Integer, gatetype) = map(T, 0.5randn(nout) / sqrt(nin))

_initial_biases{T <: AbstractFloat}(::Type{T}, nout::Integer) = ones(T, nout)

"""
Construct a new gate which projects to `node_out`.  Each node projecting to this gate should have `n` outputs.

`gatetype` should be one of:
    ALL     fully connected
    SAME    one-to-one connections
    ELSE    setdiff(ALL, SAME)
    FIXED   no learning allowed
    RANDOM  randomly connected (placeholder for future function)

All nodes and gates can be given a tag (Symbol) to identify/find in the network.
"""
function gate!{T}(node_out::Node{T}, n::Integer, gatetype::GateType = ALL;
                  tag = gensym("gate"),
                  w = _initial_weights(T, n, node_out.n, gatetype))
    # construct the state (depends on connection type)
    #   TODO: initialize w properly... not zeros
    dump(gatetype)
    w = isa(w, Function) ? w(n, node_out.n) : w
    if isa(w, AbstractVector)
        w = Diagonal(w)
    end
    state = GateState(n, w)

    # construct the connection
    g = Gate(n, gatetype, Node[], node_out, state, tag)

    # add gate reference to node_out
    push!(node_out.gates_in, g)

    g
end

function gate!(circuit::Circuit, args...; kw...)
    gate!(circuit.nodes[1], args...; kw...)
end

# ------------------------------------------------------------------------------------


"""
Project a connection from nodes_in --> gate --> node_out, or add nodes_in to the projection list.
    Asserts: all(node -> node.n == gate.n, nodes_in)

`gatetype` should be one of:
    ALL     fully connected
    SAME    one-to-one connections
    ELSE    setdiff(ALL, SAME)
    FIXED   no learning allowed
    RANDOM  randomly connected (placeholder for future function)

All nodes and gates can be given a tag (Symbol) to identify/find in the network.
"""
function project!(nodes_in::AbstractVector, g::Gate)
    if !all(node -> node.n == g.n, nodes_in)
        @show nodes_in g
        error("Size mismatch in projecting nodes to gate.")
    end

    # add gate references to nodes_in
    for node in nodes_in
        push!(gates_out(node), g)
    end
    g
end

function project!(nodes_in::AbstractVector, node_out::AbstractNode, gatetype::GateType = ALL; kw...)
    # construct the gate
    g = gate!(node_out, nodes_in[1].n, gatetype; kw...)
    g.nodes_in = nodes_in

    # project to the gate
    project!(nodes_in, g)
end

# convenience when only one node_in
function project!(node_in::AbstractNode, args...; kw...)
    project!([node_in], args...; kw...)
end

# ------------------------------------------------------------------------------------


