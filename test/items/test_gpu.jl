# Universal (backend-agnostic) GPU tests for the SurfaceletsAdaptExt extension.
#
# Everything in the 3-D stack is FFT plus a pointwise broadcast, so the device
# path needs no kernels — only `Adapt.adapt_structure` methods that move the
# buffers and replan the FFTs.  Loading `Adapt` together with a backend triggers
# the extension.
#
# GPUEnv enumerates the backends worth trying on the current host.  These items
# need one with a FFTW-compatible FFT library, because `plan_rfft` otherwise
# falls back to an AbstractFFTs generic that indexes elementwise and trips the
# scalar-indexing guard.  That rules out JLArrays, which ships no FFT
# implementation at all, and Metal / oneAPI / OpenCL, which expose their FFTs
# through their own APIs rather than the AbstractFFTs interface.  CUDA and
# AMDGPU qualify; on a host with neither, every item below is vacuously true.

@testmodule GPUBackends begin
    using GPUEnv
    using Adapt
    export fft_backends, Adapt

    # `test/runtests.jl` activates the GPU overlay before the runner starts, but a
    # test item can also be run on its own — an IDE test runner, or a language
    # server's test tooling — and that starts in the plain test environment where
    # no backend package is loadable and `gpu_backends()` comes back empty.
    # Activating here keeps the items self-contained either way; when the overlay
    # already exists (`persist = true` writes it to `test/gpu_env`) this only
    # loads it.
    isempty(gpu_backends(; include_jlarrays = false)) &&
        GPUEnv.activate(persist = true, include_jlarrays = false)

    # All test data below is Float32, so every qualifying backend can run the
    # same test bodies.
    const fft_backends = gpu_backends(; include_jlarrays = false, supports_fftw = true)
end

@testitem "GPU Adapt extension is loaded" tags = [:gpu] setup = [GPUBackends] begin
    using Adapt
    @test Base.get_extension(Surfacelets, :SurfaceletsAdaptExt) !== nothing
end


@testitem "GPU 3-D hourglass filter bank matches CPU" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    const Adapt = GPUBackends.Adapt
    Random.seed!(200)
    dims = (16, 16, 16)
    x = randn(Float32, dims...)
    cpu = make_hourglass(dims, Float32; order = 6)
    ref = hourglass_decompose(x, cpu)
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, x)
        fb = Adapt.adapt(typeof(xg), cpu)
        s = hourglass_decompose(xg, fb)
        for i in 1:3
            @test maximum(abs, Array(s[i]) .- ref[i]) < 1.0f-4
        end
        @test maximum(abs, Array(hourglass_reconstruct(s, fb)) .- x) < 1.0f-4
    end
end

@testitem "GPU 3-D directional filter bank matches CPU" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    const Adapt = GPUBackends.Adapt
    Random.seed!(201)
    dims = (16, 16, 16)
    x = randn(Float32, dims...)
    cpu = make_ndfb(dims, Float32; levels = 2, order = 6)
    ref = ndfb_decompose(x, cpu)
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, x)
        fb = Adapt.adapt(typeof(xg), cpu)
        s = ndfb_decompose(xg, fb)
        @test length(s) == ndfb_nsubbands(fb)
        for i in eachindex(s)
            @test maximum(abs, Array(s[i]) .- ref[i]) < 1.0f-4
        end
        @test maximum(abs, Array(ndfb_reconstruct(s, fb)) .- x) < 1.0f-4
    end
end

@testitem "GPU 3-D NSCT matches CPU" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    const Adapt = GPUBackends.Adapt
    Random.seed!(202)
    dims = (16, 16, 16)
    x = randn(Float32, dims...)
    p = NSSurfaceletParams(J = 2, L_array = [1, 1], order = 6)
    ws_cpu = make_ns_surfacelet_workspace(p, dims, Float32)
    ref = ns_surfacelet_forward(x, p; workspace = ws_cpu)
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, x)
        # The workspace holds real and complex tables, so adapt to the bare
        # array type; its buffers and plans are rebuilt on the device.
        ws = Adapt.adapt(backend.array_type, ws_cpu)
        c = ns_surfacelet_forward(xg, p; workspace = ws)
        @test maximum(abs, Array(c.coarse) .- ref.coarse) < 1.0f-4
        for j in 1:2, k in eachindex(c.subbands[j])
            @test maximum(abs, Array(c.subbands[j][k]) .- ref.subbands[j][k]) < 1.0f-4
        end
        rec = ns_surfacelet_inverse(c, p; workspace = ws)
        @test maximum(abs, Array(rec) .- x) < 1.0f-4
        # Streaming works on the device too.
        total = Ref(0.0f0)
        ns_surfacelet_forward(xg, p; workspace = ws) do j, k, band
            total[] += sum(abs2, Array(band))
        end
        @test total[] > 0
    end
end

@testitem "GPU Adapt.adapt_structure for NSSurfacelet coefficients" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    const Adapt = GPUBackends.Adapt
    Random.seed!(203)
    dims = (16, 16, 16)
    x = randn(Float32, dims...)
    p = NSSurfaceletParams(J = 1, L_array = [1], order = 6)
    c = ns_surfacelet_forward(x, p)
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, x)
        dev = Adapt.adapt(typeof(xg), c)
        host = Adapt.adapt(Array, dev)
        @test host.coarse == c.coarse
        @test all(k -> host.subbands[1][k] == c.subbands[1][k], eachindex(c.subbands[1]))
    end
end

@testitem "GPU surfacelet matches CPU" tags = [:gpu, :surfacelet] setup = [GPUBackends] begin
    using GPUEnv, Random
    const Adapt = GPUBackends.Adapt
    Random.seed!(204)
    dims = (48, 48, 48)
    x = randn(Float32, dims...)
    p = SurfaceletParams(J = 2, L_array = [1, 2], precision = Float32)
    # Planning evaluates every filter slice point by point, so it happens once
    # on the host; the device plan is an adapt of it, never a rebuild.
    pl_cpu = make_surfacelet(p, dims, Float32)
    ref = surfacelet_forward(x, p; plan = pl_cpu)
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, x)
        # Adapt to the bare array type, not `typeof(xg)`: the plan holds real,
        # complex and integer tables, so an element-typed destination cannot fit.
        pl = Adapt.adapt(backend.array_type, pl_cpu)
        c = surfacelet_forward(xg, p; plan = pl)
        @test size(c.coarse) == size(ref.coarse)
        @test maximum(abs, Array(c.coarse) .- ref.coarse) < 1.0f-4
        for j in 1:2, k in eachindex(c.subbands[j])
            @test size(c.subbands[j][k]) == size(ref.subbands[j][k])
            @test maximum(abs, Array(c.subbands[j][k]) .- ref.subbands[j][k]) < 1.0f-4
        end
        rec = surfacelet_inverse(c, p; plan = pl)
        @test maximum(abs, Array(rec) .- x) < 1.0f-4
        # Coefficients adapt back and forth.
        host = Adapt.adapt(Array, c)
        @test host.coarse == Array(c.coarse)
        dev = Adapt.adapt(backend.array_type, ref)
        @test Array(dev.subbands[1][1]) == ref.subbands[1][1]
    end
end

@testitem "GPU surfacelet needs an explicit plan" tags = [:gpu, :surfacelet] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(205)
    dims = (24, 24, 24)
    p = SurfaceletParams(J = 1, L_array = [1])
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, randn(Float32, dims...))
        @test_throws ArgumentError surfacelet_forward(xg, p)
    end
end

@testitem "GPU STI surfacelet matches CPU" tags = [:gpu, :surfacelet] setup = [GPUBackends] begin
    using GPUEnv, Random
    const Adapt = GPUBackends.Adapt
    Random.seed!(206)
    dims = (32, 32, 32)
    x = randn(Float32, dims...)
    p = STISurfaceletParams(J = 2, L_array = [1, 2], precision = Float32)
    pl_cpu = make_sti_surfacelet(p, dims, Float32)
    ref = sti_surfacelet_forward(x, p; plan = pl_cpu)
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, x)
        pl = Adapt.adapt(backend.array_type, pl_cpu)
        c = sti_surfacelet_forward(xg, p; plan = pl)
        @test maximum(abs, Array(c.coarse) .- ref.coarse) < 1.0f-4
        for j in 1:2, k in eachindex(c.subbands[j])
            @test maximum(abs, Array(c.subbands[j][k]) .- ref.subbands[j][k]) < 1.0f-4
        end
        @test maximum(abs, Array(sti_surfacelet_inverse(c, p; plan = pl)) .- x) < 1.0f-4
        @test_throws ArgumentError sti_surfacelet_forward(xg, p)
    end
end
