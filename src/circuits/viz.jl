
import Graphs, GraphLayout, FixedSizeArrays 

"""
create x/y lists of edge coordinates given x/y lists of vertex coordinates
  todo: add this to Plots
"""
function graph_edge_xy(adjmat, x, y)
    n = length(x)
    @assert length(y) == n
    edgex, edgey = zeros(0), zeros(0)
    for i=1:n, j=1:n
        if adjmat[i,j]
            append!(edgex, [x[i], x[j], NaN])
            append!(edgey, [y[i], y[j], NaN])
        end
    end
    edgex, edgey
end


"""
Make a Graphs.Graph from the Circuit
"""
function create_graph(net::Circuit)
    n = length(net.nodes)

    # map layer to index
    idxmap = Dict([(node,i) for (i,node) in enumerate(net)])

    # build the graph
    g = Graphs.simple_graph(n)
    for (i,node) in enumerate(net)
        for gate in node.gates_out
            j = idxmap[gate.node_out]
            Graphs.add_edge!(g, i, j)
        end
    end
    g
end


# # this code will take a start and end point and return points on the bezier curve between them
function get_curve_from_centers(c1::P2, c2::P2, yoffset::Real)
    offset = P2(0, yoffset)
    BezierCurve(c1 + offset, c2 - offset)
end

function get_circuit_points(net::Circuit; yoffset = -0.2)

    # populate vectors with the nodes
    n = length(net)
    x = 2rand(n)-1
    y = collect(1.:n)
    tags = Symbol[node.tag for node in net]
    types = fill(Node, n)
    
    # now add the gate info
    for (i,node) in enumerate(net), gate in node.gates_in
        push!(x, x[i])
        push!(y, y[i] + yoffset)
        push!(tags, gate.tag)
        push!(types, Gate)
    end

    x, y, tags, types
end



function Plots._apply_recipe(d::Dict, net::Circuit; kw...)

    # get the layout coordinates of the vertices
    g = create_graph(net)
    adjmat = Graphs.adjacency_matrix(g)

    # if x/y wasn't set manually, then use the spring layout
    x, y = if haskey(d, :x) && haskey(d, :y)
        d[:x], d[:y]
    else
        GraphLayout.layout_spring_adj(adjmat)
    end

    # get the coordinates of the edges
    # TODO: have this return midpoints of the lines as well?  (for annotating connections)
    edgex, edgey = graph_edge_xy(adjmat, x, y)

    # setup... set defaults
    get!(d, :grid, false)
    get!(d, :markersize, 20)
    get!(d, :label, ["edges" "nodes"])
    get!(d, :xlims, (-1.5,1.5))
    get!(d, :ylims, (-1.5,1.5))

    # node tags
    get!(d, :annotation, [text(node.tag) for node in net])

    # if !haskey(d, :annotation)
    #     ann = Array(Any, 1, 2)
    #     ann[1] = nothing
    #     ann[2] = [text(l.tag) for l in net]
    #     d[:annotation] = ann
    # end

    d[:linetype] = [:path :scatter]
    d[:markershape] = [:none get(d, :markershape, :rect)]

    # return the args
    Any[edgex, x], Any[edgey, y]
end
