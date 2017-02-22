### Construction of loop expressions in type domain
### This is the fallback implementation for AbstractArrays


### TODO: dispatch to choose back-end. For now, split this into dtensorop

# @dtop _[i, j] = B[i, k] * C[k, j]
#  ==
#  @top _.chunks[i,j] = Map((chunkB, chunkC) -> @top(_[i, j] = chunkB[i, k] * chunkC[k, j]), @top B.chunks[i, k] C.chunks[k, j]
#

include("tensorop.jl")

function equivalent_chunks(X::Iter)
    # An iterator on the chunks of iterators
    # TODO: handle IterConsts
    Iter(map(c -> Thunk(x -> Iter(x, X.idx), c), chunks(X.A)), X.idx)
end

function equivalent_chunks(X::Map)
    let f = X.f
        Map((args...) -> Thunk((x...) -> Map(f, x), args...),
            map(equivalent_chunks, X.Xs))
    end
end

function equivalent_chunks{dim}(X::Reduce{dim})
    @show dim
    let f = X.f
        # Reduce each chunk first
        reduced_chunks = map(c -> Thunk(x -> Reduce(dim(), f, x), c), equivalent_chunks(X.X))

        # reduce the chunks array
        Reduce(dim(), (x,y) -> Thunk((p, q) -> Reduce(dim(), f, DimCat(dim(), p, q)), x, y),
            reduced_chunks, Thunk(()->nothing)) # must be made tree reduce
    end
end

function equivalent_chunks(itr::TensorOp)
    TensorOp(equivalent_chunks(itr.lhs), equivalent_chunks(itr.rhs))
end

function dtop!(t::TensorOp)
    chunks = top!(equivalent_chunks(t))
    chunksA = chunks(t.lhs.A)
    chunksA = map(c -> Thunk(c -> top!(c)), chunksA)
    t.lhs.A
end

macro dtop(expr, reductions=:nothing)
    :(dtop!(@lower $expr $reductions))
end

using Dagger
import Dagger.chunks

function chunks(arr::Dagger.ComputedArray)
    chunks(arr.result)
end

let
    A = rand(Blocks(2,2), 4,4); B = rand(Blocks(2,2), 4,4); C = rand(Blocks(2,2), 4,4)
    A,B,C = map(compute, [A,B,C])
    D = map(identity, A)
    D = compute(D)

    @dtop A[i,j] = B[i,k]*C[k,j]
    map(gather, chunks(A))
    @test gather(D) == gather(A)
    @test gather(A) == gather(B)*gather(C)
end
