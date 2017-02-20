

using MacroTools
using Base.Test

import Base: eltype

include("util.jl")

immutable Iter{A, I}
    A::A
    idx::I # Tuple of Union{IterSym, IterConst}
end
eltype{A}(itr::Iter{A}) = eltype(A)

immutable IterSym{d} end

immutable IterConst{T}
    val::T
end


immutable Map{F, Ts}
    f::F
    Xs::Ts # Tuple of Union{Iter, ConstArg}
end
output_type(f::Function, Ts) = get(Base.return_types(f, Ts), 1, Any)
eltype{F,Ts}(itr::Map{F, Ts}) = output_type(itr.f, map(eltype, itr.Xs))

immutable ConstArg{T}
    val::T
end

immutable Reduce{dim, F, T, E}
    f::F
    X::T
    empty::E
end

function Reduce{I<:IterSym,F,T}(dim::I, f::F, X::T)
    ident = reduce_identity(f, eltype(X))
    Reduce{I,F,T, typeof(ident)}(f,X,ident)
end

eltype{dim,F,T,E}(itr::Reduce{dim, F,T,E}) = output_type(itr.f, (E, eltype(itr.X)))
reduce_identity{T}(f::typeof(+), ::Type{T}) = zero(T)
reduce_identity{T}(f::typeof(*), ::Type{T}) = one(T)

immutable TensorOp{L<:Iter,R}
    lhs::L
    rhs::R
end

# Lowering a tensor operation expression to Iter, Map, Reduce

function lower_index(idx, only_symbols=false)
    if isa(idx, Symbol)
        IterSym{idx}()
    else
        if only_symbols
            throw(ArgumentError("Got $idx instead of a symbol"))
        end
        :(IterConst($idx))
    end
end

function lower_maps(expr)
    @match expr begin
        A_[idx__] => :(Iter($A, ($(map(lower_index, idx)...),)))
        f_(arg_)   => :(Map($f, ($(lower_maps(arg)),)))
        f_(args__)  => :(Map($f, ($(reduce(vcat, [lower_maps(x) for x in args])...),)))
        x_ => :(ConstArg($x))
    end
end

function reduction_functions(reductions)
    @match reductions begin
        (i_=>f_) => Dict(i => f)
        [R__] => reduce(merge, map(reduction_functions, R))
        nothing => Dict()
        _ => error("Invalid reduction spec")
    end
end

function lower(expr, reductions)
    lhs, rhs = @match expr begin
        (lhs_ = rhs_) => lhs, rhs
        _ => error("Expression is not of the form LHS = RHS")
    end
    lidxs = indicesinvolved(lhs)
    ridxs = indicesinvolved(rhs)

    # which indices iterate over which dimension of the input
    # idxdims = index_dim_map(ridxs) # TODO: use this to verify correct dimensions
    lowered_maps = lower_maps(rhs)

    # which indices are reduced over
    reduceddims = setdiff(flatten(cdr.(ridxs)), flatten(cdr.(lidxs)))
    reduce_dict = reduction_functions(reductions)
    rhs_lowered = reduce(lowered_maps, reduceddims) do ex, idx
        :(Reduce($(lower_index(idx, true)), $(get(reduce_dict, idx, +)), $ex))
    end

    :(TensorOp($(lower_maps(lhs)), $rhs_lowered))
end


"""
    top!(t::TensorOp)

Perform a tensor operation
"""
function top!(x) x end

macro top(expr, reductions=:nothing)
    :(top!($(lower(expr, reductions)))) |> esc
end

