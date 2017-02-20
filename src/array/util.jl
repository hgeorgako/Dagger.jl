using Combinatorics

car(x::Pair) = x.first
cdr(x::Pair) = x.second
flatten(xs) = reduce(vcat, vec.(xs))
function indicesinvolved(expr)
    @match expr begin
        A_[idx__] => [(A => idx)]
        f_(ex__)  => reduce(vcat, [indicesinvolved(x) for x in ex])
        _ => []
    end
end

let
    @test indicesinvolved(:(A[i,j,k])) == [:A=>Any[:i,:j,:k]]
    @test indicesinvolved(:(A[i,j,k]+B[x,y,z])) == [:A=>Any[:i,:j,:k], :B=>[:x,:y,:z]]
    @test indicesinvolved(:(A[i,j,k] |> f)) == [:A=>Any[:i,:j,:k]]
end

function index_dim_map(xs)
    iters = Dict()
    for (tensor, idxs) in xs
        for (i, idx) in enumerate(idxs)
            dims = Base.@get! iters idx []
            push!(dims, (tensor, i))
        end
    end
    iters
end
