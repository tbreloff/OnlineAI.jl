
# See: http://www.overcomplete.net/papers/nn2012.pdf  (Derek Monner 2012)

# Gist: LSTM-g (Generalized Long Short Term Memory) is a more general version of LSTM
#       which can be easily used in alternative network configurations, including 
#       hierarchically stacking.  Connections are gated, as opposed to the activations.

# Important methods: GatedLayers can be `connect`ed together, and `gate`d by another layer.

# ------------------------------------------------------------------------------------

export
    layer,
    gate!,
    ALL,
    SAME,
    ELSE


# ------------------------------------------------------------------------------------

"Holds the current state of the layer"
immutable State{T}
    Σ::Vector{T}
    a::Vector{T}
end
State{T}(::Type{T}, n::Integer) = State(zeros(T,n), zeros(T,n))

"""
This is the core object... the LSTM-g layer.  We track connections in and out,
as well as any connections that this layer gates.
"""
immutable GatedLayer{T}
    n::Int            # number of nodes in the layer
    conn_in::Vector   # connections coming in
    conn_out::Vector  # connections going out
    gates::Vector     # all connections that this layer gates
    state::State{T}   # current state of the layer
    tag::Symbol
end


# ------------------------------------------------------------------------------------

"Holds a weight and bias for state calculation"
immutable ConnectionCalc{W <: AbstractArray, B <: AbstractArray}
    w::W
    b::B
end


@enum ConnectionType ALL SAME ELSE

"""
Connect one `GatedLayer` to another.  May have a gate as well.
"""
type GatedConnection{T, CC <: ConnectionCalc}
    conn_type::ConnectionType
    layer_from::GatedLayer{T}
    layer_to::GatedLayer{T}
    calc::CC
    gate::Nullable{GatedLayer{T}}
    tag::Symbol
end


# ------------------------------------------------------------------------------------

# Constructors


function GatedLayer{T}(::Type{T}, n::Integer, tag::Symbol = gensym("layer"))
    GatedLayer(n, GatedConnection[], GatedConnection[], GatedConnection[], State(T, n), tag)
end

layer(n::Integer; tag::Symbol = gensym("layer")) = GatedLayer(Float64, n, tag)


# function GatedConnection{T}(conn_type::ConnectionType, layer_from::GatedLayer{T}, layer_to::GatedLayer{T}, calc::ConnectionCalc)
#     GatedConnection(conn_type, layer_from, layer_to, calc, Nullable{GatedLayer{T}}())
# end


# ------------------------------------------------------------------------------------


function Base.connect{T}(layer_from::GatedLayer{T}, layer_to::GatedLayer{T}, conn_type::ConnectionType = ALL; tag = gensym("conn"))
    numfrom = layer_from.n
    numto = layer_to.n

    # construct the calc (depends on connection type)
    #   TODO: initialize w/b properly... not zeros
    w = conn_type == ALL ? zeros(T, numto, numfrom) : zeros(T, numto)
    b = zeros(T, numto)
    calc = ConnectionCalc(w, b)

    # construct the connection
    conn = GatedConnection(conn_type, layer_from, layer_to, calc, Nullable{GatedLayer{T}}(), tag)

    # add connection reference to layers
    push!(layer_from.conn_out, conn)
    push!(layer_to.conn_in, conn)

    conn
end


"Makes `by_layer` gate the connection `conn`"
function gate!(conn::GatedConnection, by_layer::GatedLayer)
    # don't overwrite the gate
    @assert isnull(conn.gate)

    # add the references
    conn.gate = Nullable(by_layer)
    push!(by_layer.gates, conn)

    conn
end


