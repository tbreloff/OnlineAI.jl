
# See: http://www.overcomplete.net/papers/nn2012.pdf  (Derek Monner 2012)

# Gist: LSTM-g (Generalized Long Short Term Memory) is a more general version of LSTM
#       which can be easily used in alternative network configurations, including 
#       hierarchically stacking.  Connections are gated, as opposed to the activations.

# Important methods: GatedLayers can be `connect`ed together, and `gate`d by another layer.

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
    write(io, "Node{ tag=$(l.tag) n=$(l.n) in=$(stringtags(l.gates_in)) out=$(stringtags(l.gates_out)) state=$(l.state)}")
end

# ------------------------------------------------------------------------------------

"Holds a weight and bias for state calculation"
immutable GateState{T, W <: AbstractArray{T}}
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
type Gate{T, GS <: GateState{T}}
    n::Int                  # number of cells of nodes projecting in
    gatetype::GateType      # connectivity type
    nodes_in::Vector{Node{T}}
    node_out::Node{T}
    state::GS
    tag::Symbol
end

function Base.show(io::IO, c::Gate)
    write(io, "Gate{ tag=$(c.tag) type=$(c.gatetype) from=$(c.nodes_in.tag) to=$(c.nodes_out.tag)}")
end



# ------------------------------------------------------------------------------------

"""
Reference to a set of connected nodes, defined by the input/output nodes.
"""
immutable Circuit{T}
    nodes::Vector{Node{T}}
    nodemap::Dict{Symbol,Node{T}}
    gatemap::Dict{Symbol,Gate{T}}
end

function Circuit{T}(nodes::AVec{Node{T}}, gates = [])
    # first add missing gates
    gates = Set(gates)
    for node in nodes, gate in node.gates_in
        push!(gates, gate)
    end

    nodemap = Dict([(node.tag, node) for node in nodes])
    gatemap = Dict([(gate.tag, gate) for gate in gates])
    Circuit(nodes, nodemap, gatemap)
end

# TODO: constructor which takes inputlayer/outputlayer and initializes nodes with a proper ordering (traversing connection graph)

Base.start(net::Circuit) = 1
Base.done(net::Circuit, state::Int) = state > length(net.nodes)
Base.next(net::Circuit, state::Int) = (net.nodes[state], state+1)

Base.size(net::Circuit) = size(net.nodes)
Base.length(net::Circuit) = length(net.nodes)

# ------------------------------------------------------------------------------------

# Constructors


function Node{T}(::Type{T}, n::Integer, activation::Activation = IdentityActivation(); tag::Symbol = gensym("layer"))
    Node(n, activation, Gate[], Gate[], NodeState(T, n), tag)
end
Node(args...; kw...) = Node(Float64, args...; kw...)


# ------------------------------------------------------------------------------------


function project!{T}(nodes_in::AVec{Node{T}}, node_out::Node{T}, gatetype::GateType = ALL; tag = gensym("conn"))
    # TODO: assert n is the same for all of nodes_in
    n = nodes_in[1].n
    numto = node_out.n

    # construct the state (depends on connection type)
    #   TODO: initialize w properly... not zeros
    w = gatetype == ALL ? zeros(T, numto, n) : zeros(T, numto)
    state = GateState(n, w)

    # construct the connection
    g = Gate(gatetype, nodes_in, nodes_out, state, tag)

    # add gate reference to nodes
    push!(nodes_in.gates_out, g)
    push!(node_out.gates_in, g)

    g
end


"Makes `by_layer` gate the connection `conn`"
function gate!(g::Gate, by_layer::Node)
    # don't overwrite the gate
    @assert isnull(g.gate)

    # add the references
    g.gate = Nullable(by_layer)
    push!(by_layer.gates, g)

    g
end


