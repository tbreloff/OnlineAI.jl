
# See for inspiration: http://www.overcomplete.net/papers/nn2012.pdf  (Derek Monner 2012)
# Gist: LSTM-g (Generalized Long Short Term Memory) is a more general version of LSTM
#       which can be easily used in alternative network configurations, including 
#       hierarchically stacking.  Connections are gated, as opposed to the activations.

# Neural Circuits:
#   `Node`s can be `project!`ed towards `Gate`s, which then project to a single `Node`.

# ------------------------------------------------------------------------------------

export
    AbstractNode,
    Gate,
    Node,
    Circuit,

    nodes,
    circuits,
    gate!,
    project!,

    @circuit_str,
    @gates_str,

    ALL,
    SAME,
    ELSE,
    FIXED,
    RANDOM

abstract AbstractNode <: NeuralNetLayer

# ------------------------------------------------------------------------------------

"Holds the current state of the layer"
immutable NodeState{T, PUSTATE <: ParameterUpdaterState}
    s::Vector{T}            # state of the node: sⱼ = ∑ gᴳⱼ + bⱼ
    y::Vector{T}            # output of the node: yⱼ = f(sⱼ)
    δ::Vector{T}            # error responsibility: ∂(E(t-1) + E(t)) / ∂sⱼ  ==  ϕⱼ + ψⱼ
    ϕ::Vector{T}            # error responsibility for feedforward connections only
    ψ::Vector{T}            # error responsibility for recurrent connections only
    f_ratio_prev::Vector{T}  # used in computation of ψ: f_ratio(t-1) = f'(s(t-1)) / y(t-1)
    ∇::Vector{T}            # \nabla:   online gradient for weight matrix: ∇ⱼᵢ(τ) = γ (∇ⱼᵢ(τ-1) + ψⱼ εᵢ(t-1)) + ϕⱼ εᵢ(t)
    b::Vector{T}            # bias
    b_states::Vector{PUSTATE}  # bias states
end

function NodeState{T}(::Type{T}, n::Integer)
    NodeState(ones(T, n))
end

function NodeState{T}(b::AbstractVector{T})
    n = length(b)
    NodeState(zeros(T,n),
              zeros(T,n),
              zeros(T,n),
              zeros(T,n),
              zeros(T,n),
              zeros(T,n),
              zeros(T,n),
              b,
              ParameterUpdaterState(n)) # TODO: handle updaters different from current_updater?
end

"""
This is the core object... the Neural Circuit Node.  We track gates projecting in and projections out to gates.
A Node is equivalent to a layer in an classic artificial neural network, with 1 to many cells representing the 
individual neurons.
"""
immutable Node{T, A <: Mapping} <: AbstractNode
    n::Int            # number of nodes in the layer
    mapping::A
    gates_in::Vector  # connections coming in
    gates_out::Vector # connections going out
    state::NodeState{T}   # current state of the layer
    tag::Symbol
end

Node(node::AbstractNode, args...; kw...) = node
function Node{T}(::Type{T}, n::Integer, mapping::Mapping = IdentityActivation(); tag::Symbol = gensym("node"))
    Node(n, mapping, Gate[], Gate[], NodeState(T, n), tag)
end
Node(args...; kw...) = Node(Float64, args...; kw...)

gates_in(node::Node) = node.gates_in
gates_out(node::Node) = node.gates_out

stringtags(v::AbstractVector) = string("[", join([string(c.tag) for c in v], ", "), "]")

function Base.show(io::IO, l::Node; prefix = "")
    write(io, prefix, "Node{ tag=$(l.tag) n=$(l.n) f=$(typeof(l.mapping)) in=$(stringtags(l.gates_in)) out=$(stringtags(l.gates_out))}")
end

# ------------------------------------------------------------------------------------

"Holds a weight and bias for state calculation"
type GateState{T, W <: AbstractArray, PUSTATE <: ParameterUpdaterState}
    g::Vector{T}    # the state of the gate: g(τ) = w * ∏y
    ε::Vector{T}    # \epsilon: eligibility trace for weight update:  ε(t) = x(t) = ∏yᵢ
    ε_prev::Vector{T} # ε(t-1)
    ∇::W            # \nabla:   online gradient for weight matrix: ∇ⱼᵢ(τ) = γ (∇ⱼᵢ(τ-1) + ψⱼ εᵢ(t-1)) + ϕⱼ εᵢ(t)
    w::W            # weight matrix (may be diagonal for SAME or sparse for RANDOM)
    w_states::Matrix{PUSTATE}  # states of the gradient calc
end

function GateState{T}(w::AbstractMatrix{T})
    n, m = size(w)
    GateState(zeros(T,n),
              zeros(T,m),
              zeros(T,m),
              w,
              fill!(similar(w), 0),
              ParameterUpdaterState(n, m)) # TODO: handle updaters different from current_updater?
end

# TODO: need to be able to pass parameters for random connectivity!
@enum GateType ALL SAME ELSE FIXED RANDOM

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

function Base.show(io::IO, g::Gate)
    write(io, "Gate{ tag=$(g.tag) n=$(g.n) type=$(g.gatetype) from=$(stringtags(g.nodes_in)) to=$(g.node_out.tag)}")
end



# ------------------------------------------------------------------------------------

"""
Reference to a set of connected nodes, defined by the input/output nodes.
"""
type Circuit <: AbstractNode
    n::Int
    nodes::Vector{AbstractNode}
    nodemap::Dict{Symbol,AbstractNode}
    gatemap::Dict{Symbol,Gate}
    updater::ParameterUpdater
    mloss::ModelLoss
    ploss::ParameterLoss
    tag::Symbol
end

function Circuit(nodes::AbstractVector, gates = [];
                 updater::ParameterUpdater = current_updater(),
                 mloss::ModelLoss = current_mloss(),
                 ploss::ModelLoss = current_ploss(),
                 tag::Symbol = gensym("circuit"))
    # first add missing gates
    gates = Set(gates)
    for node in nodes, gate in gates_in(node)
        push!(gates, gate)
    end

    nodemap = Dict{Symbol,AbstractNode}([(node.tag, node) for node in nodes])
    gatemap = Dict{Symbol,Gate}([(gate.tag, gate) for gate in gates])
    Circuit(nodes[1].n, nodes, nodemap, gatemap, updater, mloss, ploss, tag)
end

# TODO: constructor which takes inputlayer/outputlayer and initializes nodes with a proper ordering (traversing connection graph)

Base.start(net::Circuit) = 1
Base.done(net::Circuit, state::Int) = state > length(net.nodes)
Base.next(net::Circuit, state::Int) = (net.nodes[state], state+1)

Base.size(net::Circuit) = size(net.nodes)
Base.length(net::Circuit) = length(net.nodes)

Base.getindex(net::Circuit, i::Integer) = net.nodes[i]
Base.getindex(net::Circuit, s::AbstractString) = net[symbol(s)]
function Base.getindex(net::Circuit, s::Symbol)
    try
        net.nodemap[s]
    catch
        net.gatemap[s]
    end
end

function Base.show(io::IO, net::Circuit; prefix = "")
    write(io, prefix, "Circuit{ tag=$(net.tag) n=$(net.n)\n$prefix Nodes:\n")
    for node in net.nodes
        write(io, " "^4)
        show(io, node, prefix = prefix*" "^4)
        write(io, "\n")
        for gate in gates_in(node)
            write(io, prefix, " "^6)
            show(io, gate)
            write(io, "\n")
        end
    end
    write(io, prefix, "}")
end

function findindex(net::Circuit, node::AbstractNode)
    for (i,tmpnode) in enumerate(net)
        if tmpnode === node
            return i
        elseif isa(tmpnode, Circuit)
            try
                findindex(tmpnode, node)
                return i
            end
        end
    end
    error("couldn't find node: $node")
end

nodes(net::Circuit) = filter(n -> isa(n, Node), net.nodes)
circuits(net::Circuit) = filter(n -> isa(n, Circuit), net.nodes)
gates_in(net::Circuit) = gates_in(net.nodes[1])
gates_out(net::Circuit) = gates_out(net.nodes[end])

# ------------------------------------------------------------------------------------

include("gates.jl")
include("macros.jl")
include("learn.jl")
include("viz.jl")

