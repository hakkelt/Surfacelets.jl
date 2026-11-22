@testitem "NSSurfacelet — interface conformance" tags = [:ns_surfacelet] begin
    using MultiscaleTransformsBase.Testing: test_transform
    using Random
    t = NSSurfacelet(scales = 2, direction_levels = [1, 2], bernstein_order = 6)
    test_transform(t, randn(Xoshiro(1), 16, 12, 20); wrong_size = (8, 8, 8))
    test_transform(t, randn(Xoshiro(2), ComplexF64, 16, 12, 20); exact_real_linear = true, wrong_size = (8, 8, 8))
    test_transform(t, randn(Xoshiro(3), Float32, 16, 16, 16); rtol = 1.0f-4)
    test_transform(NSSurfacelet(scales = 1, direction_levels = [1], bernstein_order = 4, tight = false), randn(Xoshiro(4), 12, 12, 12))
    # Volumes whose byte size is not a multiple of the FFT alignment: the
    # subbands of the stacked coefficients start at misaligned addresses.
    test_transform(t, randn(Xoshiro(5), 9, 7, 5))
    test_transform(t, randn(Xoshiro(6), ComplexF32, 15, 15, 9); exact_real_linear = true, rtol = 1.0f-4)
end

@testitem "Surfacelet — interface conformance" tags = [:surfacelet] begin
    using MultiscaleTransformsBase.Testing: test_transform
    using Random
    t = Surfacelet(scales = 2, direction_levels = [1, 2])
    x = randn(Xoshiro(7), 48, 24, 36)
    z = randn(Xoshiro(8), ComplexF64, 48, 24, 36)
    # Calls without a workspace plan with FFTW.ESTIMATE, which uses the wisdom
    # that FFTW.MEASURE planning (here, inside the conformance tests) leaves
    # behind.  Plan once up front, so that every call of the exact
    # real-linearity check plans alike.
    make_workspace(t, z)
    test_transform(t, x; wrong_size = (24, 24, 24))
    test_transform(t, z; exact_real_linear = true, wrong_size = (24, 24, 24))
    # Subband offsets in the coefficient vector that break the FFT alignment.
    u = Surfacelet(scales = 1, direction_levels = [2], pyramid_ratio = 1)
    y = randn(Xoshiro(9), Float32, 18, 18, 18)
    @test any(r -> ((first(r) - 1) * sizeof(Float32)) % 16 != 0, similar_coefficients(u, y).layout.ranges)
    make_workspace(u, complex(y))
    test_transform(u, y; rtol = 1.0f-4)
    test_transform(u, complex.(y, reverse(y; dims = 2)); exact_real_linear = true, rtol = 1.0f-4)
end

@testitem "coefficient formats" tags = [:ns_surfacelet, :surfacelet] begin
    using Random
    x = randn(Xoshiro(10), 16, 12, 20)
    t = NSSurfacelet(scales = 2, direction_levels = [1, 2], bernstein_order = 6)
    c = analyze(t, x)
    n = nsubbands(t, size(x))
    @test c isa Array{Float64, 4} && size(c) == (size(x)..., n)
    @test subband(t, c, 3) === view(c, :, :, :, 3)
    @test size(similar_coefficients(t, rand(Int8, 4, 4, 4))) == (4, 4, 4, 1 + 6 + 12)
    @test eltype(similar_coefficients(t, rand(Int8, 4, 4, 4))) == Float64
    @test_throws DimensionMismatch analyze!(similar(c, size(x)..., n - 1), t, x)
    @test_throws "three axes" similar_coefficients(t, rand(4, 4))

    s = Surfacelet(scales = 2, direction_levels = [1, 2])
    xs = randn(Xoshiro(11), 48, 48, 48)
    cs = analyze(s, xs)
    @test cs isa SubbandVector{Float64}
    @test length(cs) == round(Int, surfacelet_redundancy(s, size(xs)) * length(xs))
    @test size(subband(s, cs, 1)) == (16, 16, 16)
    @test all(i -> subband(s, cs, i) == subband(cs, i), 1:nsubbands(s, size(xs)))
    @test_throws DimensionMismatch analyze!(similar_coefficients(Surfacelet(scales = 2, direction_levels = [2, 2]), xs), s, xs)
    @test_throws "element type" analyze!(SubbandVector(zeros(Float32, length(cs)), cs.layout), s, xs)
end

@testitem "make_workspace — shared keywords" tags = [:ns_surfacelet, :surfacelet] begin
    using FFTW, Random
    # An irfft plan is a scaled brfft plan.
    estimate(p) = hasproperty(p, :flags) ? p.flags & FFTW.ESTIMATE != 0 : estimate(p.p)
    nt = Threads.nthreads()
    cases = (
        (NSSurfacelet(scales = 2, direction_levels = [1, 2], bernstein_order = 6), (16, 12, 20)),
        (Surfacelet(scales = 2, direction_levels = [1, 2]), (48, 24, 36)),
    )
    for (t, dims) in cases, Td in (Float64, ComplexF64)
        x = randn(Xoshiro(21), Td, dims...)
        ns = t isa NSSurfacelet
        plans(ws) = ns ? (ws.engine.pfwd, ws.engine.pinv) : (ws.pfwd[1], ws.pbwd[1], ws.cfwd[2], ws.work[1].hg[1].fwd)
        # flags reach the host FFT plans (FFTW.MEASURE is 0, so test the ESTIMATE bit).
        # MEASURE planning leaves wisdom that later ESTIMATE plans use, so it comes
        # first: every ESTIMATE workspace below then plans alike.
        @test !any(estimate, plans(make_workspace(t, Td, dims; prewarm = false)))
        ref = make_workspace(t, x; flags = FFTW.ESTIMATE, prewarm = false)
        @test all(estimate, plans(ref))
        c = analyze(ref, x)
        # tasks: `nothing` is one per thread, `1` a serial workspace.
        for f in (() -> make_workspace(t, x; flags = FFTW.ESTIMATE), () -> make_workspace(t, Td, dims; flags = FFTW.ESTIMATE))
            ws = f()
            if ns
                @test ws.tasks === nothing && length(ws.engine.buffers) == nt
            else
                @test ws.tasks == nt
            end
        end
        one = make_workspace(t, Td, dims; flags = FFTW.ESTIMATE, tasks = Int32(1))
        if ns
            @test length(one.engine.buffers) == 1
            # A streaming call without a workspace builds a serial one.
            @test length(Surfacelets._ns_workspace_for(nothing, t, x; tasks = 1).engine.buffers) == 1
        else
            @test one.tasks == 1 && one.pfwd[1] === one.pfwd[2]
        end
        @test analyze(one, x) == c
        @test_throws "tasks must be ≥ 1" make_workspace(t, x; tasks = 0)
        @test_throws "tasks must be ≥ 1" make_workspace(t, Td, dims; tasks = 0)
        # tune is accepted and changes nothing; prewarm leaves no state behind.
        for kw in ((; tune = true), (; prewarm = true), (; tune = true, prewarm = false))
            ws = make_workspace(t, x; flags = FFTW.ESTIMATE, kw...)
            @test analyze(ws, x) == c
            @test synthesize(ws, c) == synthesize(ref, c)
            @test analyze(make_workspace(t, Td, dims; flags = FFTW.ESTIMATE, kw...), x) == c
        end
        @test_throws MethodError make_workspace(t, x; no_such_keyword = true)
        @test_throws MethodError make_workspace(t, Td, dims; no_such_keyword = true)
        # The shared keywords do not hide the size error.
        @test_throws "three axes" make_workspace(t, Td, (16, 16); flags = FFTW.ESTIMATE, tasks = 1, tune = false, prewarm = false)
    end
end

@testitem "workspaces and banks show compactly" tags = [:ns_surfacelet, :surfacelet] begin
    using Surfacelets: make_hourglass, make_ndfb
    ws = make_workspace(NSSurfacelet(scales = 1), Float32, (16, 16, 16); prewarm = false, tasks = 2)
    sw = make_workspace(Surfacelet(scales = 1, direction_levels = [1]), Float64, (12, 12, 12); prewarm = false)
    h = make_hourglass(Float64, (8, 8, 8); bernstein_order = 6)
    n = make_ndfb(Float64, (8, 8, 8); levels = 1)
    @test startswith(repr(ws), "NSSurfaceletWorkspace{Float32}(16×16×16, NSSurfacelet(scales = 1")
    @test endswith(repr(ws), "tasks = 2)")
    @test startswith(repr(sw), "SurfaceletWorkspace{Float64}(12×12×12, Surfacelet(scales = 1")
    @test repr(h) == "HourglassFB{Float64}((8, 8, 8), bernstein_order = 6, halfband_order = 1, tight = true)"
    @test startswith(repr(n), "NDFB{Float64}((8, 8, 8), levels = ")
    for x in (ws, sw, h, n)
        @test length(repr(MIME"text/plain"(), x)) < 300
    end
end
