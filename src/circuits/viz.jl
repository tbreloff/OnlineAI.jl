
import Graphs, GraphLayout

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
