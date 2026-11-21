# Workspaces for device arrays are built on the host and moved with `Adapt`,
# whose methods live in the `SurfaceletsAdaptExt` extension.

# `ws`, moved to the array type of `x` unless `x` is a host array.
_workspace_like(::Array, ws) = ws
_workspace_like(x::AbstractArray, ws) = _is_host(x) ? ws : _adapt_workspace(x, ws)

# Views and wrappers of host arrays stay on the host.
_is_host(x::Array) = true
_is_host(x::AbstractArray) = (p = parent(x)) !== x && _is_host(p)

function _adapt_workspace(x::AbstractArray, ws)
    ext = Base.get_extension(@__MODULE__, :SurfaceletsAdaptExt)
    ext === nothing && throw(ArgumentError("a workspace for a $(typeof(x)) needs Adapt.jl loaded"))
    # The bare array type (`CuArray`): the workspace holds real, complex and
    # integer tables, which an element-typed destination cannot all fit.
    return ext.adapt_workspace(typeof(x).name.wrapper, ws)
end

# One analysis and synthesis of zeros with `ws`, in arrays like `proto` (host or
# device), so that the first call with `ws` is compiled and finds its buffers
# touched.
function _prewarm!(ws::AbstractTransformWorkspace, proto::AbstractArray)
    x = fill!(similar(proto, eltype(ws), size(ws)), zero(eltype(ws)))
    c = similar_coefficients(transform(ws), x)
    analyze!(c, ws, x)
    synthesize!(x, ws, c)
    return ws
end
