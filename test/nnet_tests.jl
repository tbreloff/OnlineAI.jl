
module NNetTest

using OnlineAI, FactCheck

srand(1)

function xor_data()
    inputs = [0 0; 0 1; 1 0; 1 1]
    targets = float(sum(inputs,2) .== 1)

    # all sets are the same
    inputs = inputs .- mean(inputs,1)
    DataPoints(inputs, targets)
end


function testxor{LAYER}(::Type{LAYER};
        hiddenLayerNodes = [4],
        hiddenMapping = SoftsignMapping(),
        finalMapping = SigmoidMapping(),
        params = NetParams(),
        solverParams = SolverParams(maxiter=100000),
        inputTransformer = IdentityTransformer(),
        doPretrain = false)

    # all xor inputs and results
    inputs = [0 0; 0 1; 1 0; 1 1]
    targets = float(sum(inputs,2) .== 1)

    # all sets are the same
    inputs = inputs .- mean(inputs,1)
    data = DataPoints(inputs, targets)
    sampler = SimpleSampler(data)

    # hiddenLayerNodes = [2]
    net = buildRegressionNet(
        LAYER,
        ncols(inputs),
        ncols(targets),
        hiddenLayerNodes;
        hiddenMapping = hiddenMapping,
        finalMapping = finalMapping,
        params = params,
        solverParams = solverParams,
        inputTransformer = inputTransformer
    )
    show(net)

    if doPretrain
        pretrain(net, sampler, sampler)
    end

    stats = solve!(net, sampler, sampler)

    output = vec(predict(net, inputs))
    for (o, d) in zip(output, data)
        println("Result: input=$(d.x) target=$(d.y) output=$o")
    end

    net, output, stats
end

for LAYER in (Layer, NormalizedLayer)
    facts("NeuralNet{$LAYER}") do

        atol = 0.05
        solverParams = SolverParams(
            maxiter=50000,
            erroriter=10000,
            minerror=1e-3,
            plotiter=-1,
            plotfields=Symbol[:x, :xhat, :β, :α, :δy, :y, :w, :b, :δΣ, :Σ, :a]
        )

        updater = AdaMaxUpdater()
        net, output, stats = testxor(
            LAYER,
            params=NetParams(updater=updater, mloss=L2DistLoss()),
            solverParams=solverParams,
            doPretrain=false
        )
        @fact output --> roughly([0., 1., 1., 0.], atol=atol)

        net, output, stats = testxor(
            LAYER,
            params=NetParams(updater=AdagradUpdater(), mloss=L2DistLoss()),
            solverParams=solverParams
        )
        @fact output --> roughly([0., 1., 1., 0.], atol=atol)

        net, output, stats = testxor(
            LAYER,
            params=NetParams(updater=AdaMaxUpdater(), mloss=CrossentropyLoss()),
            finalMapping=SigmoidMapping(),
            solverParams=solverParams
        )
        @fact output --> roughly([0., 1., 1., 0.], atol=atol)

        solverParams.maxiter = 300000
        net, output, stats = testxor(
            LAYER,
            params=NetParams(updater=AdaMaxUpdater(), mloss=CrossentropyLoss(),
            dropout=Dropout(1.0,0.9)),
            finalMapping=SigmoidMapping(),
            solverParams=solverParams,
            hiddenLayerNodes = [6,6,6]
        )
        @fact output --> roughly([0., 1., 1., 0.], atol=atol)

    end # facts
end


function test_pretrain(; solve=true, pretr=true, netparams=NetParams(), kwargs...)
    data = xor_data()
    sampler = StratifiedSampler(data)

    # nin = 1; f = x->x[1:1]
    nin = 2; f = nop
    net = buildRegressionNet(
        Layer, nin,1,[2]; params=netparams,
        solverParams=SolverParams(maxiter=10000),
        inputTransformer=f
    )
    if pretr
        pretrain(net, sampler, sampler; kwargs...)
    end

    if solve
        solve!(net, sampler, sampler)
    end

    net, Float64[predict(net,d.x)[1] for d in data]
end


end # module
nn = NNetTest
