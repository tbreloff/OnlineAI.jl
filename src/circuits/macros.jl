
const _activation_names = Dict(
    "identity"  => IdentityActivation,
    "sigmoid"   => SigmoidActivation,
    "tanh"      => TanhActivation,
    "softsign"  => SoftsignActivation,
    "relu"      => ReLUActivation,
    "lrelu"     => LReLUActivation,
    "softmax"   => SoftmaxActivation,
    )

strip_comment(str) = strip(split(str, '#')[1])

"split up the string `str` by the character(s)/string(s) `chars`, strip each token, and filter empty tokens"
tokenize(str, chars) = map(strip, split(str, chars, keep=false))
tokenize_commas(str) = tokenize(str, [' ', ','])

nop(x) = x

"might need to wrap in quotes"
index_val(str::AbstractString) = try; parse(Int, str); catch; str; end
index_val(str::AbstractString, f::Function) = try; parse(Int, str); catch; f(str); end

# ------------------------------------------------------------------------------------

"""
Convenience macro to build a set of nodes into an ordered Neural Circuit.
Each row defines a node.  The first value should be an integer which is the 
number of output cells for that node.  The rest will greedily apply to other
node features:
    
    - An activation name/alias will set the node's activation function.
        - note: default activation is IdentityActivation
    - Other symbols will set the tag.
    - A vector-type or Function will initialize the bias vector.

Note: comments (anything after `#`) and all spacing will be ignored

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
    expr = :(Circuit(AbstractNode[]))
    constructor_list = expr.args[2].args

    # parse out string into vector of args for each node
    lines = split(strip(str), '\n')
    for l in lines
        args = split(strip_comment(l))

        # n = number of cells in this node
        n = index_val(args[1], symbol)

        # if it's an activation, override IdentityActivation, otherwise assume it's a tag
        activation = IdentityActivation
        tag = string(gensym("node"))
        for arg in args[2:end]
            if haskey(_activation_names, arg)
                activation = _activation_names[arg]
            else
                tag = arg
            end
        end

        # create the Node
        nodeexpr = :(Node($n, $activation(); tag = symbol($tag)))

        # add Node definition to the constructor_list
        push!(constructor_list, nodeexpr)
    end
    esc(expr)
end

# ------------------------------------------------------------------------------------


"""
Convenience macro to construct gates and define projections from nodes --> gates --> nodes.
First line should be the circuit.  Subsequent lines have the format: `<nodes_in> --> <nodes_out>`.
Since every gate has exactly one `node_out`, there will be one gate created for every node in `nodes_out`.

    - Extra arguments should go after a semicolon
    - Nodes can be Int or Symbol (tag)... they will be passed directly to Base.getindex
    - Gate types accepted: ALL, SAME, ELSE, FIXED, RANDOM
    - Vector, Matrix, or Function will be used to initialize the weight array
    - Anything else will be applied as a tag

Note: comments (anything after `#`) and all spacing will be ignored

Example:

```
gates\"\"\"
    lstm
    1   --> 2,3,5               # input projections
    1,2 --> 3                   # input gate
    3,4 --> 4; FIXED, w=ones(5) # forget gate
    4   --> 2,3,5               # peephole connections
    4,5 --> 6                   # output gate
end
\"\"\"
```
"""
macro gates_str(str)

    # set up the expression
    expr = Expr(:block)

    # parse out string into vector of args for each node
    lines = split(strip(str), '\n')
    lines = map(strip_comment, lines)

    # this is the circuit to add to
    circuit = symbol(lines[1])

    for l in lines[2:end]

        # gotta have a mapping in order to process this line
        contains(l, "-->") || continue

        # grab kw args if any
        mapping, args = if ';' in l
            tokenize(l, ";")
        else
            l, ""
        end

        # handle extra arguments greedily
        gatetype = :ALL
        kw = Dict()
        # dump(parse(args), 10)
        argexpr = parse(args)
        if isa(argexpr, Expr) && argexpr.head == :tuple
            for arg in argexpr.args
                if arg in [:ALL, :SAME, :ELSE, :FIXED, :RANDOM]
                    gatetype = arg
                elseif isa(arg, Expr)
                    # assume it's an initial weight
                    kw[:w] = esc(arg)
                elseif isa(arg, Symbol)
                    # assume it's a tag
                    kw[:tag] = arg
                else
                    warn("arg not processed in gates_str macro: $arg")
                    # try
                    #     # keyword arg
                    #     k,v = tokenize(arg, "=")
                    #     kw[symbol(k)] = parse(v)
                    # catch
                    #     # assume it's a tag
                    #     kw[:tag] = symbol(arg)
                    # end
                end
            end
        end

        # process the mapping
        nodes_in, nodes_out = map(tokenize_commas, tokenize(mapping, "-->"))
        for node_out in nodes_out

            # build an expression to project from nodes_in to node_out
            ex = :(project!(AbstractNode[], $(esc(circuit))[$(index_val(node_out))], $gatetype))

            # add the kw
            for (k,v) in kw
                push!(ex.args, Expr(:kw, k, v))
            end

            # add the nodes_in
            ninargs = ex.args[2].args
            for node_in in nodes_in
                push!(ninargs, :($(esc(circuit))[$(index_val(node_in))]))
            end

            push!(expr.args, ex)
        end
    end
    push!(expr.args, esc(circuit))
    # dump(expr, 10)
    @show expr
    expr
end
