
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

"Append points from a BezierCurve to an existing list, adding an arrow at the end, plus a closing NaN"
function add_curve_points!(pts::AVec, curve::BezierCurve)
    append!(pts, curve_points(curve))

    # add the arrow head
    lastpt = pts[end]
    sz = 0.005
    append!(pts, [lastpt + P2(-sz,-2sz), lastpt + P2(sz,-2sz), lastpt])

    # add a NaN point to separate line segments
    push!(pts, P2(NaN,NaN))
end

# define a rectangle shape for representing Nodes
const _box_w = 1.0
const _box_h = 0.2
const _box = Shape([
        (-_box_w, -_box_h),
        (-_box_w,  _box_h),
        ( _box_w,  _box_h),
        ( _box_w, -_box_h)
    ])

# -------------------------------------------------------------------

function Plots.plot(net::Circuit; kw...)
    d = Dict(kw)

    # collect some parameters
    noffset = P2(0, get(d, :noffset, 0.02))
    goffset = P2(0, get(d, :goffset, 0.0))
    lineargs = get(d, :line, (2, 0.7))
    gatediff = get(d, :gatediff, P2(0, -0.05))

    # get the positions of the nodes
    n = length(net)
    node_pts = get(d, :node_pts) do
        nodex = get(d, :node_x) do
            n < 2 ? rand(n) : vcat(0.5, rand(n-2), 0.5)
        end
        nodey = linspace(0, 1, n)
        P2[_ for _ in zip(nodex,nodey)]
    end

    n_pts = P2[]
    c_pts = P2[]
    g_pts = P2[]
    # g2n_pts = P2[]
    n2g_pts = P2[]
    n2g_recur_pts = P2[]

    # collect points for gate positions and the curves
    for (i,node_out) in enumerate(net)

        # populate node/circuit poits lists
        push!(isa(node_out, Circuit) ? c_pts : n_pts, node_pts[i])

        # this gives the x-offset for putting gates side-by-side
        gatesin = gates_in(node_out)
        ng = length(gatesin)
        xrng = if ng > 1
            linspace(-1, 1, length(gatesin)) * sqrt(length(gatesin)-1) * 0.02
        else
            0:0
        end

        # compute the points and add the curves for each gate projecting in
        for (j,g) in enumerate(gatesin)

            # calc and add the gate point
            pt = node_pts[i] + gatediff + P2(xrng[j],0)
            push!(g_pts, pt)

            # # create a directed curve from the gate to the node_out
            # add_curve_points!(g2n_pts, directed_curve(pt + goffset, node_pts[i] - noffset))

            # add curve from nodes_in to gates
            for node_in in g.nodes_in
                k = findindex(net, node_in)
                add_curve_points!(k < i ? n2g_pts : n2g_recur_pts,
                                  directed_curve(node_pts[k] + noffset, pt - goffset))
            end
        end
    end

    # set up the plot
    plt = plot(grid = false,
               xticks = nothing,
               yticks = nothing,
               xlims = (-0.1,1.1),
               ylims = (-0.1,1.1))

    # nodes-to-gates curves
    if !isempty(n2g_pts)
        plot!(n2g_pts, lab = "forward", line = lineargs)
    end

    # recurrent nodes-to-gates 
    if !isempty(n2g_recur_pts)
        plot!(n2g_recur_pts, line = lineargs, lab = "recurrent")
    end

    # # gates-to-nodes curves
    # if !isempty(g2n_pts)
    #     plot!(g2n_pts, lab = "gate to node", line = lineargs)
    # end

    # nodes
    if !isempty(n_pts)
        ms = get(d, :ms, 50)
        scatter!(n_pts,
                ann = [text(node.tag,12) for node in nodes(net)],
                lab = "nodes",
                m = (ms, _box, 0.6, :cyan))
    end

    # circuits
    if !isempty(c_pts)
        ms = get(d, :ms, 50)
        scatter!(c_pts,
                ann = [text(c.tag,12) for c in circuits(net)],
                lab = "circuits",
                m = (ms, :hex, 0.6, :pink))
    end

    # gates
    if !isempty(g_pts)
        txt = text("Π", :white, 5)
        scatter!(g_pts, lab = "gates",
                 m = (6,:black, 0.7),
                 ann = fill(txt, length(g_pts)))
    end

    plt
end


# -------------------------------------------------------------------

# function Plots._apply_recipe(d::Dict, net::Circuit; kw...)

#     # get the layout coordinates of the vertices
#     g = create_graph(net)
#     adjmat = Graphs.adjacency_matrix(g)

#     # if x/y wasn't set manually, then use the spring layout
#     x, y = if haskey(d, :x) && haskey(d, :y)
#         d[:x], d[:y]
#     else
#         GraphLayout.layout_spring_adj(adjmat)
#     end

#     # get the coordinates of the edges
#     # TODO: have this return midpoints of the lines as well?  (for annotating connections)
#     edgex, edgey = graph_edge_xy(adjmat, x, y)

#     # setup... set defaults
#     get!(d, :grid, false)
#     get!(d, :markersize, 50)
#     get!(d, :label, ["edges" "nodes"])
#     get!(d, :xlims, (-1.5,1.5))
#     get!(d, :ylims, (-1.5,1.5))

#     # node tags
#     get!(d, :annotation, [text(node.tag) for node in net])

#     # if !haskey(d, :annotation)
#     #     ann = Array(Any, 1, 2)
#     #     ann[1] = nothing
#     #     ann[2] = [text(l.tag) for l in net]
#     #     d[:annotation] = ann
#     # end

#     d[:linetype] = [:path :scatter]
#     d[:markershape] = [:none get(d, :markershape, _node_rect)]

#     # return the args
#     Any[edgex, x], Any[edgey, y]
# end
