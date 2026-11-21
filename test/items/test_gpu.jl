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
    isempty(gpu_backends(; include_cpu_backends = false)) &&
        GPUEnv.activate(persist = true, include_cpu_backends = false)

    # All test data below is Float32, so every qualifying backend can run the
    # same test bodies.
    const fft_backends = gpu_backends(; include_cpu_backends = false, supports_fftw = true)
end

@testitem "GPU Adapt extension is loaded" tags = [:gpu] setup = [GPUBackends] begin
    using Adapt
    @test Base.get_extension(Surfacelets, :SurfaceletsAdaptExt) !== nothing
end


@testitem "GPU 3-D hourglass filter bank matches CPU" tags = [:gpu] setup = [GPUBackends] begin
    using Surfacelets: make_hourglass, hourglass_decompose, hourglass_reconstruct
    using GPUEnv, Random
    const Adapt = GPUBackends.Adapt
    Random.seed!(200)
    dims = (16, 16, 16)
    x = randn(Float32, dims...)
    cpu = make_hourglass(dims, Float32; bernstein_order = 6)
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
    using Surfacelets: make_ndfb, ndfb_nsubbands, ndfb_decompose, ndfb_reconstruct
    using GPUEnv, Random
    const Adapt = GPUBackends.Adapt
    Random.seed!(201)
    dims = (16, 16, 16)
    x = randn(Float32, dims...)
    cpu = make_ndfb(dims, Float32; levels = 2, bernstein_order = 6)
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

@testitem "GPU nonsubsampled surfacelet matches CPU" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    const Adapt = GPUBackends.Adapt
    Random.seed!(202)
    dims = (16, 16, 16)
    x = randn(Float32, dims...)
    t = NSSurfacelet(scales = 2, direction_levels = [1, 1], bernstein_order = 6)
    n = nsubbands(t, dims)
    ws_cpu = make_workspace(t, Float32, dims)
    ref = analyze(t, x; workspace = ws_cpu)
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, x)
        # The workspace holds real and complex tables, so adapt to the bare
        # array type; its buffers and plans are rebuilt on the device.
        ws = Adapt.adapt(backend.array_type, ws_cpu)
        c = analyze(t, xg; workspace = ws)
        for i in 1:n
            @test maximum(abs, Array(subband(t, c, i)) .- subband(t, ref, i)) < 1.0f-4
        end
        rec = synthesize(t, c; workspace = ws)
        @test maximum(abs, Array(rec) .- x) < 1.0f-4
        # A workspace made from a device array lives on the device.
        wd = make_workspace(t, xg)
        @test maximum(abs, Array(synthesize(t, analyze(t, xg; workspace = wd); workspace = wd)) .- x) < 1.0f-4
        # Streaming works on the device too.
        total = Ref(0.0f0)
        analyze(t, xg; workspace = ws) do i, band
            total[] += sum(abs2, Array(band))
        end
        @test total[] ≈ sum(abs2, x) rtol = 1.0f-3
    end
end

@testitem "GPU nonsubsampled surfacelet: complex data" tags = [:gpu] setup = [GPUBackends] begin
    using GPUEnv, Random
    const Adapt = GPUBackends.Adapt
    Random.seed!(203)
    dims = (16, 16, 16)
    z = randn(ComplexF32, dims...)
    t = NSSurfacelet(scales = 1, direction_levels = [1], bernstein_order = 6)
    ref = analyze(t, z)
    for backend in GPUBackends.fft_backends
        zg = to_gpu(backend, z)
        ws = make_workspace(t, zg)
        c = analyze(t, zg; workspace = ws)
        @test all(i -> maximum(abs, Array(subband(t, c, i)) .- subband(t, ref, i)) < 1.0f-4, 1:nsubbands(t, dims))
        @test maximum(abs, Array(synthesize(t, c; workspace = ws)) .- z) < 1.0f-4
        # Streaming runs both channels of a subband together on the device.
        analyze(t, zg; workspace = ws) do i, band
            @test maximum(abs, Array(band) .- subband(t, ref, i)) < 1.0f-4
        end
        y = similar(zg)
        synthesize!(i -> subband(t, c, i), y, t; workspace = ws)
        @test maximum(abs, Array(y) .- z) < 1.0f-4
    end
end

@testitem "GPU surfacelet matches CPU" tags = [:gpu, :surfacelet] setup = [GPUBackends] begin
    using GPUEnv, Random
    const Adapt = GPUBackends.Adapt
    Random.seed!(204)
    dims = (48, 48, 48)
    x = randn(Float32, dims...)
    t = Surfacelet(scales = 2, direction_levels = [1, 2])
    n = nsubbands(t, dims)
    # Building the workspace evaluates every filter slice point by point, so it
    # happens once on the host; the device workspace is an adapt of it.
    ws_cpu = make_workspace(t, Float32, dims)
    ref = analyze(t, x; workspace = ws_cpu)
    for backend in GPUBackends.fft_backends
        xg = to_gpu(backend, x)
        # Adapt to the bare array type, not `typeof(xg)`: the workspace holds
        # real, complex and integer tables, so an element-typed destination
        # cannot fit.
        ws = Adapt.adapt(backend.array_type, ws_cpu)
        c = analyze(t, xg; workspace = ws)
        for i in 1:n
            @test size(subband(t, c, i)) == size(subband(t, ref, i))
            @test maximum(abs, Array(subband(t, c, i)) .- subband(t, ref, i)) < 1.0f-4
        end
        rec = synthesize(t, c; workspace = ws)
        @test maximum(abs, Array(rec) .- x) < 1.0f-4
    end
end

@testitem "GPU surfacelet without a workspace, and complex data" tags = [:gpu, :surfacelet] setup = [GPUBackends] begin
    using GPUEnv, Random
    Random.seed!(205)
    dims = (24, 24, 24)
    t = Surfacelet(scales = 1, direction_levels = [1])
    z = randn(ComplexF32, dims...)
    ref = analyze(t, z)
    for backend in GPUBackends.fft_backends
        zg = to_gpu(backend, z)
        # Without a workspace, one is built on the host and moved to the device.
        c = analyze(t, zg)
        @test all(i -> maximum(abs, Array(subband(t, c, i)) .- subband(t, ref, i)) < 1.0f-4, 1:nsubbands(t, dims))
        @test maximum(abs, Array(synthesize(t, c)) .- z) < 1.0f-4
    end
end
