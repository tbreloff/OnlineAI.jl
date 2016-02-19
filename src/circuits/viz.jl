
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

const _rect_w = 2.0
const _rect_h = 0.5
const _node_rect = Shape([(-_rect_w,_rect_h),(_rect_w,_rect_h),(_rect_w,-_rect_h),(-_rect_w,-_rect_h)])

# # # this code will take a start and end point and return points on the bezier curve between them
# function get_curve_from_centers(c1::P2, c2::P2)
#     BezierCurve(c1, c2) |> points
# end

# function get_node_points(net::Circuit)
#     n = length(net)
#     x = 2rand(n)-1
#     y = collect(1.:n)
#     tags = Symbol[node.tag for node in net]
#     types = fill(Node, n)
# end

# function get_node_points(net::Circuit; yoffset = -0.2)    
#     # now add the gate info
#     for (i,node) in enumerate(net), gate in node.gates_in
#         push!(x, x[i])
#         push!(y, y[i] + yoffset)
#         push!(tags, gate.tag)
#         push!(types, Gate)
#     end

#     x, y, tags, types
# end

function add_curve_points!(pts::AVec, curve::BezierCurve)
    append!(pts, points(curve))
    lastpt = pts[end]
    sz = 0.02
    append!(pts, [lastpt + P2(-sz,-2sz), lastpt + P2(sz,-2sz), lastpt])
    push!(pts, NaN)
end

const _box_w = 1.0
const _box_h = 0.2
const _box = Shape([
        (-_box_w, -_box_h),
        (-_box_w,  _box_h),
        ( _box_w,  _box_h),
        ( _box_w, -_box_h)
    ])

function Plots.plot(net::Circuit; kw...)
    d = Dict(kw)
    noffset = P2(0, get(d, :noffset, 0.13))
    goffset = P2(0, get(d, :goffset, 0.1))

    n = length(net)
    node_pts = get(d, :node_pts) do
        nodex = 2rand(n)-1
        nodey = linspace(-1, 1, n)
        P2[_ for _ in zip(nodex,nodey)]
    end

    # collect point for gate positions and the curves
    g2n_pts = P2[]
    n2g_pts = P2[]
    gate_pts = P2[]
    for (i,node) in enumerate(net), g in node.gates_in
        # # calc pt as average of connected nodes
        # pt = if haskey(d, :gatediff)
        #     node_pts[i] + d[:gatediff]
        # else
        #     mean(node_pts[i], node_pts[i],
        #           [node_pts[findindex(net,nodein)] for nodein in g.nodes_in]...)
        # end
        pt = node_pts[i] + get(d, :gatediff, P2(0, -0.4))
        push!(gate_pts, pt)

        # create a bezier curve from the gate to the node_out
        # complete the line segment with a NaN
        add_curve_points!(g2n_pts, BezierCurve(pt + goffset, node_pts[i] - noffset))

        # add curve from nodes_in to gates
        for nodein in g.nodes_in
            add_curve_points!(n2g_pts, BezierCurve(node_pts[findindex(net,nodein)] + noffset, pt - goffset))
        end
    end

    g2n_pts = g2n_pts[1:end-1]
    n2g_pts = n2g_pts[1:end-1]

    w = get(d, :w, 2)

    plot(n2g_pts,
        grid = false,
        xticks = nothing,
        yticks = nothing,
        lab = "node --> gate",
        line = (w, 0.7),
        xlims = (-1.5,1.5),
        ylims = (-1.5,1.5))

    plot!(g2n_pts, lab = "gate --> node", line = (w, 0.7))

    ms = get(d, :ms, 50)
    scatter!(node_pts,
            ann = [node.tag for node in net],
            lab = "nodes",
            m = (ms, _box, 0.6, :cyan))

    scatter!(gate_pts, lab = "gates", m = (10,:black, 0.6))

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
    get!(d, :markersize, 50)
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
    d[:markershape] = [:none get(d, :markershape, _node_rect)]

    # return the args
    Any[edgex, x], Any[edgey, y]
end
