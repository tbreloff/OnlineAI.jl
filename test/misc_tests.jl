
using OnlineAI, FactCheck

n = 10_000
x = rand(n, 10)
y = rand(n, 1)

nin = ncols(x)
nout = ncols(y)

gradientModel = AdadeltaModel(1e-6, 0.95, 1e-6)
dropout = Dropout()
costModel = L2CostModel()
params = NetParams(gradientModel, dropout, costModel)



