using MacroTools
using Base.Test

include("util.jl")

function indicesinvolved(expr)
    @match expr begin
        A_[idx__] => [(A => idx)]
        f_(ex__) =>   reduce(vcat, [indicesinvolved(x) for x in ex])
        _ => []
    end
end

let
    @test indicesinvolved(:(A[i,j,k])) == [:A=>Any[:i,:j,:k]]
    @test indicesinvolved(:(A[i,j,k]+B[x,y,z])) == [:A=>Any[:i,:j,:k], :B=>[:x,:y,:z]]
    @test indicesinvolved(:(A[i,j,k] |> f)) == [:A=>Any[:i,:j,:k]]
end

function indexdims(xs)
    iters = Dict()
    for (tensor, idxs) in xs
        for (i, idx) in enumerate(idxs)
            dims = Base.@get! iters[idx] []
            push!(dims, (tensor, i))
        end
    end
    iters
end

#=
function aligneddims(xs)
    # for every pair of tensors in the expression,
    map(combinations(xs, 2)) do x
        ltensor, rtensor = car.(x)

        # find the pairs of dimensions that are equal
        l,r = collect.(enumerate.(cdr.(x)))
        [x[2] == y[2] ? (ltensor, x[1]) => (rtensor, y[1]) : nothing
             for x in l, y in r] |> pairs -> filter(x->x!=nothing, pairs)

    end |> flatten
end

let
    function testaligndim(x, y)
        z = indicesinvolved(x) |> aligneddims
        @test z == y
    end

    testaligndim(:(A[i,j] * B[j,k]), Any[(:A, 2) => (:B,1)])
    testaligndim(:(A[j,k,l] * B[j,k,l]),
                 Pair.(tuple.([:A], [1,2,3]), tuple.([:B], [1,2,3])))
    testaligndim(:(A[j,k,l] * B[l,j,k]),
                 Pair.(tuple.([:A], [3,1,2]), tuple.([:B], [1,2,3])))
end
=#

function tensorop(expr)
    lhs, rhs = @match expr begin
        (lhs_ = rhs_) => lhs, rhs
    end
    lidxs = indicesinvolved(lhs)
    ridxs = indicesinvolved(rhs)

    aligned = aligneddims(ridxs)
    # check if these dimensions are equal

    reduceddims = setdiff(flatten(cdr.(ridxs)), flatten(cdr.(lidxs)))

    # first we generate the expression which reduces
    nested_loops(reduceddims)
end


