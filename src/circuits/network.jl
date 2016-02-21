
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

    @circuit_str,

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
A Node is equivalent to a layer in an classic artificial neural network, with 1 to many cells representing the 
individual neurons.
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
    write(io, "Node{ tag=$(l.tag) n=$(l.n) f=$(typeof(l.activation)) in=$(stringtags(l.gates_in)) out=$(stringtags(l.gates_out))}")
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

const _activation_names = Dict(
    "identity"  => "IdentityActivation",
    "sigmoid"   => "SigmoidActivation",
    "tanh"      => "TanhActivation",
    "softsign"  => "SoftsignActivation",
    "relu"      => "ReLUActivation",
    "lrelu"     => "LReLUActivation",
    "softmax"   => "SoftmaxActivation",
    )

"""
Convenience macro to build a set of nodes into an ordered Neural Circuit.
Each row defines a node.  The first value should be an integer which is the 
number of output cells for that node.  The rest will greedily apply to other
node features:
    
    - An activation name/alias will set the node's activation function.
        - note: default activation is IdentityActivation
    - Other symbols will set the tag.
    - A vector-type or Function will initialize the bias vector.

Example:

```
lstm = circuit\"\"\"
    3 in
    5 inputgate sigmoid
    5 forgetgate sigmoid
    5 memorycell
    5 forgetgate sigmoid
    1 output
\"\"\"
```
"""
macro circuit_str(str)

    # set up the expression
    expr = :(Circuit(Node[]))
    constructor_list = expr.args[2].args

    # parse out string into vector of args for each node
    str2 = split(strip(str), '\n')
    lines = map(s->split(strip(s)), str2)

    for l in lines

        # n = number of cells in this node
        n = l[1]

        # if it's an activation, override IdentityActivation, otherwise assume it's a tag
        activation = "IdentityActivation"
        tagstr = ""
        for arg in l[2:end]
            if haskey(_activation_names, arg)
                activation = _activation_names[arg]
            else
                tagstr = "tag=symbol(\"$arg\")"
            end
        end

        # create the Node
        nodestr = "Node($n, $activation(); $tagstr)"

        # add Node definition to the constructor_list
        push!(constructor_list, parse(nodestr))
    end
    expr
end

# ------------------------------------------------------------------------------------

"""
Construct a new gate which projects to `node_out`.  Each node projecting to this gate should have `n` outputs.

`gatetype` should be one of:
    ALL     fully connected
    SAME    one-to-one connections
    ELSE    setdiff(ALL, SAME)
    RANDOM  randomly connected (placeholder for future function)

All nodes and gates can be given a tag (Symbol) to identify/find in the network.
"""
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

# ------------------------------------------------------------------------------------


"""
Project a connection from nodes_in --> gate --> node_out, or add nodes_in to the projection list.
    Asserts: all(node -> node.n == gate.n, nodes_in)

`gatetype` should be one of:
    ALL     fully connected
    SAME    one-to-one connections
    ELSE    setdiff(ALL, SAME)
    RANDOM  randomly connected (placeholder for future function)

All nodes and gates can be given a tag (Symbol) to identify/find in the network.
"""
function project!(nodes_in::AbstractVector, g::Gate)
    @assert all(node -> node.n == g.n, nodes_in)
    # add gate references to nodes_in
    for node in nodes_in
        push!(node.gates_out, g)
    end
    g
end

function project!(nodes_in::AbstractVector, node_out::Node, gatetype::GateType = ALL; tag = gensym("gate"))
    # construct the gate
    g = gate!(node_out, nodes_in[1].n, gatetype; tag = tag)
    g.nodes_in = nodes_in

    # project to the gate
    project!(nodes_in, g)
end

# convenience when only one node_in
function project!(node_in::Node, args...; kw...)
    project!([node_in], args...; kw...)
end





