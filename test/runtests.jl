using GPUEnv

# Build (and activate) a GPU overlay on top of this test environment.  GPUEnv
# adds any functional native backend (CUDA, AMDGPU, …) so it need not be listed
# as a test dep and the `:gpu` test items can exercise the Adapt extension.
# JLArrays is excluded: it has no FFT implementation, so `plan_rfft` falls back
# to a scalar-indexing generic and every 3-D item would run it on the host.
GPUEnv.activate(persist = true, include_jlarrays = false)

using TestItemRunner
const FILTER_PARTS = if length(ARGS) > 0
    @assert length(ARGS) == 1
    split(ARGS[1], ",")
else
    String[]
end
const FILTER_TAGS = map(p -> Symbol(p[2:end]), filter(x -> startswith(x, ":"), FILTER_PARTS))
const FILTER_NAMES = filter(x -> !startswith(x, ":"), FILTER_PARTS)

const VERB = get(ENV, "SURFACELETS_TEST_VERBOSE", "false") == "true"
const FILTER = if length(FILTER_PARTS) > 0
    ti -> begin
        run_item = any(t -> t in ti.tags, FILTER_TAGS) || any(n -> n == ti.name, FILTER_NAMES)
        if VERB && run_item
            println("Running @testitem: ", ti.name)
        end
        run_item
    end
else
    ti -> begin
        VERB && println("Running @testitem: ", ti.name)
        true
    end
end

@run_package_tests filter = FILTER
