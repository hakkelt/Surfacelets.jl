# Workspaces for device arrays are built on the host and moved with `Adapt`,
# whose methods live in the `SurfaceletsAdaptExt` extension.

# `ws`, moved to the device of `x` when the data of `x` lives on one; every
# host array (a `BitArray`, a range, a view of either, …) keeps the host workspace.
_workspace_like(x::AbstractArray, ws) = is_device_array(x) ? _adapt_workspace(x, ws) : ws

function _adapt_workspace(x::AbstractArray, ws)
    ext = Base.get_extension(@__MODULE__, :SurfaceletsAdaptExt)
    ext === nothing && throw(ArgumentError("a workspace for a $(typeof(x)) needs Adapt.jl loaded"))
    # The bare storage type (`CuArray`, also for a view of one): the workspace
    # holds real, complex and integer tables, which an element-typed destination
    # cannot all fit.
    return ext.adapt_workspace(storage_type(x), ws)
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
