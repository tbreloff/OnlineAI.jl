

typealias VecF Vector{Float64}
typealias MatF Matrix{Float64}
typealias VecI Vector{Int}
typealias MatI Matrix{Int}
typealias IntIterable Union(VecI, StepRange{Int,Int}, UnitRange{Int})


####################################################

donothing(x...) = return
nop(x) = x
returntrue(x) = true

unzip(x) = [p[1] for p in x], [p[2] for p in x]


sizes(x) = map(size, x)

mapf(functions, obj) = [f(obj) for f in functions]

####################################################


column(mat::Matrix, i::Int) = mat[:,i]
row(mat::Matrix, i::Int) = reshape(mat[i,:], size(mat,2))
column(vec::Vector, i::Int) = (i == 1 ? vec : error("Asking for column $i of a vector!"))
row(vec::Vector, i::Int) = vec[i:i]

col(matorvec, i::Int) = column(matorvec, i)

addOnesColumn(mat::MatF) = hcat(mat, ones(size(mat, 1)))
addOnesColumn(vec::VecF) = hcat(vec, ones(length(vec)))


# columns(mat::Matrix, rng::IntIterable) = copy(sub(mat, :, rng))
columns(mat::Matrix, rng::IntIterable) = mat[:,rng]
rows(mat::Matrix, rng::IntIterable) = mat[rng,:]
columns(vec::Vector, rng::IntIterable) = (rng == 1:1 ? vec : error("Asking for columns $rng of a vector!"))
rows(vec::Vector, rng::IntIterable) = vec[rng]

cols(matorvec, rng::IntIterable) = columns(matorvec, rng)

nrows(matorvec) = size(matorvec,1)
ncols(matorvec) = size(matorvec,2)

row!(M::Matrix, r::Int, v::Vector) = setrow(M, r, v)
col!(M::Matrix, c::Int, v::Vector) = setcol(M, c, v)

function setrow(M::Matrix, row::Int, v::Vector)
	for col in 1:length(v)
		M[row,col] = v[col]
	end
end

function setcol(M::Matrix, col::Int, v::Vector)
	for row in 1:length(v)
		M[row,col] = v[row]
	end
end


mat(t::DataType = Float64) = zeros(t, 0, 0)
mat(m::Matrix) = m
mat(v::Vector) = reshape(v, length(v), 1)


getPctOfInt(pct::Float64, T::Int) = round(Int, max(0., min(1., pct)) * T)

function splitRange(pct::Float64, T::Int)
	lastin = getPctOfInt(pct,T)
	1:lastin, lastin+1:T
end

function splitMatrixRows(mat::Matrix, pct::Float64)
	rng1, rng2 = splitRange(pct, nrows(mat))
	rows(mat,rng1), rows(mat,rng2)
end

function apply(f::Function, A::AbstractArray)
	for x in A
		f(x)
	end
end