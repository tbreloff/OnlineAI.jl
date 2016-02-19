
# See for inspiration: http://www.overcomplete.net/papers/nn2012.pdf  (Derek Monner 2012)
# Gist: LSTM-g (Generalized Long Short Term Memory) is a more general version of LSTM
#       which can be easily used in alternative network configurations, including 
#       hierarchically stacking.  Connections are gated, as opposed to the activations.

# Neural Circuits:
#   `Node`s can be `project!`ed towards `Gate`s, which then project to a single `Node`.

# ------------------------------------------------------------------------------------

export
    Gate,
    Node,
    Circuit,

    gate!,
    project!,

    ALL,
    SAME,
    ELSE,
    RANDOM


# ------------------------------------------------------------------------------------

"Holds the current state of the layer"
immutable NodeState{T}
    s::Vector{T}
    y::Vector{T}
    δ::Vector{T}
    b::Vector{T}
end
NodeState{T}(::Type{T}, n::Integer) = NodeState(zeros(T,n), zeros(T,n), zeros(T,n), ones(T,n))

"""
This is the core object... the Neural Circuit Node.  We track gates projecting in and projections out to gates.
"""
immutable Node{T, A <: Activation} <: NeuralNetLayer
    n::Int            # number of nodes in the layer
    activation::A
    gates_in::Vector  # connections coming in
    gates_out::Vector # connections going out
    state::NodeState{T}   # current state of the layer
    tag::Symbol
end

stringtags(v::AbstractVector) = string("[", join([string(c.tag) for c in v], ", "), "]")

function Base.show(io::IO, l::Node)
    write(io, "Node{ tag=$(l.tag) n=$(l.n) in=$(stringtags(l.gates_in)) out=$(stringtags(l.gates_out))}")
end

# ------------------------------------------------------------------------------------

"Holds a weight and bias for state calculation"
immutable GateState{T, W <: AbstractArray}
    w::W            # weight matrix (may be diagonal for SAME or sparse for RANDOM)
    ε::Vector{T}    # eligibility trace for weight update:  ε = ∏yᵢ
    ∇::Vector{T}    # online gradient: ∇(τ) = γ ∇(τ-1) + δₒᵤₜδₙε
    s::Vector{T}    # the state of the gate: s(τ) = w * ∏yᵢ
end
GateState{T}(n::Integer, w::AbstractArray{T}) = GateState(w, zeros(T,n), zeros(T,n), zeros(T,n))

# TODO: need to be able to pass parameters for random connectivity!
@enum GateType ALL SAME ELSE RANDOM

"""
Connect one `Node` to another.  May have a gate as well.
"""
type Gate{T}
    n::Int                  # number of cells of nodes projecting in
    gatetype::GateType      # connectivity type
    nodes_in::Vector
    node_out::Node{T}
    state::GateState
    tag::Symbol
end

function Base.show(io::IO, c::Gate)
    write(io, "Gate{ tag=$(c.tag) type=$(c.gatetype) from=$(stringtags(c.nodes_in)) to=$(c.node_out.tag)}")
end



# ------------------------------------------------------------------------------------

"""
Reference to a set of connected nodes, defined by the input/output nodes.
"""
immutable Circuit
    nodes::Vector{Node}
    nodemap::Dict{Symbol,Node}
    gatemap::Dict{Symbol,Gate}
end

function Circuit(nodes::AbstractVector, gates = [])
    # first add missing gates
    gates = Set(gates)
    for node in nodes, gate in node.gates_in
        push!(gates, gate)
    end

    nodemap = Dict{Symbol,Node}([(node.tag, node) for node in nodes])
    gatemap = Dict{Symbol,Gate}([(gate.tag, gate) for gate in gates])
    Circuit(nodes, nodemap, gatemap)
end

# TODO: constructor which takes inputlayer/outputlayer and initializes nodes with a proper ordering (traversing connection graph)

Base.start(net::Circuit) = 1
Base.done(net::Circuit, state::Int) = state > length(net.nodes)
Base.next(net::Circuit, state::Int) = (net.nodes[state], state+1)

Base.size(net::Circuit) = size(net.nodes)
Base.length(net::Circuit) = length(net.nodes)

function Base.show(io::IO, net::Circuit)
    write(io, "Circuit{\n  Nodes:\n")
    for node in net.nodes
        write(io, " "^4)
        show(io, node)
        write(io, "\n")
        for gate in node.gates_in
            write(io, " "^6)
            show(io, gate)
            write(io, "\n")
        end
    end
    write(io, "}")
end

function findindex(net::Circuit, node::Node)
    for (i,tmpnode) in enumerate(net)
        if tmpnode === node
            return i
        end
    end
    error("couldn't find node: $node")
end

# ------------------------------------------------------------------------------------

# Constructors


function Node{T}(::Type{T}, n::Integer, activation::Activation = IdentityActivation(); tag::Symbol = gensym("node"))
    Node(n, activation, Gate[], Gate[], NodeState(T, n), tag)
end
Node(args...; kw...) = Node(Float64, args...; kw...)


# ------------------------------------------------------------------------------------


function project!(nodes_in::AbstractVector, node_out::Node, gatetype::GateType = ALL; tag = gensym("gate"))
    # TODO: assert n is the same for all of nodes_in
    n = nodes_in[1].n

    # construct the gate
    g = gate!(node_out, n, gatetype; tag = tag)
    g.nodes_in = nodes_in

    # add gate references to nodes_in
    for node in nodes_in
        push!(node.gates_out, g)
    end

    g
end

function gate!{T}(node_out::Node{T}, n::Integer, gatetype::GateType = ALL; tag = gensym("gate"))
    # construct the state (depends on connection type)
    #   TODO: initialize w properly... not zeros
    numout = node_out.n
    w = gatetype == ALL ? zeros(T, numout, n) : zeros(T, numout)
    state = GateState(n, w)

    # construct the connection
    g = Gate(n, gatetype, Node[], node_out, state, tag)

    # add gate reference to node_out
    push!(node_out.gates_in, g)

    g
end


# "Makes `by_layer` gate the connection `conn`"
# function gate!(g::Gate, by_layer::Node)
#     # don't overwrite the gate
#     @assert isnull(g.gate)

#     # add the references
#     g.gate = Nullable(by_layer)
#     push!(by_layer.gates, g)

#     g
# end


