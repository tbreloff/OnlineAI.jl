

typealias AVecI AbstractVector{Int}
typealias AMatI AbstractMatrix{Int}
typealias VecI Vector{Int}
typealias MatI Matrix{Int}
typealias IntIterable AVec{Int}
typealias FloatIterable AVecF

# --------------------------------------------------------

immutable TransposeView{T} <: AbstractMatrix{T}
  mat::Matrix{T}
end
Base.getindex(tv::TransposeView, i::Integer, j::Integer) = tv.mat[j,i]
Base.setindex!{T}(tv::TransposeView{T}, val::T, i::Integer, j::Integer) = (tv.mat[j,i] = val)
Base.length(tv::TransposeView) = length(tv.mat)
Base.size(tv::TransposeView) = (ncols(tv.mat), nrows(tv.mat))

# import Base: *, ctranspose
# function *{T<:Real}(tv::TransposeView{T}, v::AVec{T})
#   res = similar(v, T, nrows(tv))
#   for i in 1:nrows(tv)
#     res[i] = dot(col(tv.mat, i), v)
#   end
#   res
# end

# ctranspose(tv::TransposeView) = tv.mat


####################################################

donothing(x...) = return
nop(x) = x
returntrue(x) = true

unzip(x) = [p[1] for p in x], [p[2] for p in x]


sizes(x) = map(size, x)

mapf(functions, obj) = [f(obj) for f in functions]


row(M::AMat, i::Integer) = rowvec_view(M, i)
col(M::AMat, i::Integer) = view(M, :, i)
column(M::AMat, i::Integer) = view(M, :, i)

rows(M::AMat, rng::AVecI) = view(M, rng, :)
cols(M::AMat, rng::AVecI) = view(M, :, rng)
columns(M::AMat, rng::AVecI) = view(M, :, rng)

row!{T}(M::AMat{T}, i::Integer, v::AVec{T}) = (M[i,:] = v; nothing)
col!{T}(M::AMat{T}, i::Integer, v::AVec{T}) = (M[:,i] = v; nothing)
column!{T}(M::AMat{T}, i::Integer, v::AVec{T}) = (M[:,i] = v; nothing)

function if0then1(x::Real)
  if isnan(x) || x == 0
    one(typeof(x))
  else
    x
  end
end


####################################################



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

function foreach(A::AbstractArray, f::Function, fs::Function...)
  for x in A
    f(x)
  end
  for g in fs
    foreach(A, g)
  end
  A
end


function getLinspace(n, h)
  h = n > 1 ? h : 0
  linspace(-h, h, n)
end

