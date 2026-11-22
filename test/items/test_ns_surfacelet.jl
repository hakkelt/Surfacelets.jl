@testitem "NSSurfacelet — construction, validation and equality" tags = [:ns_surfacelet] begin
    using MultiscaleTransformsBase: supports_streaming
    t = NSSurfacelet(scales = 3, direction_levels = [1, 2, 2])
    @test t.scales == 3
    @test t.direction_levels == [1, 2, 2]
    @test t == NSSurfacelet(scales = 3, direction_levels = [1, 2, 2])
    @test hash(t) == hash(NSSurfacelet(scales = 3, direction_levels = Int32[1, 2, 2]))
    # -0.0 and 0.0 are one transition: equal and hashing alike.
    @test hash(NSSurfacelet(scales = 1, transition = -0.0)) == hash(NSSurfacelet(scales = 1, transition = 0.0))
    @test length(Set([NSSurfacelet(scales = 1, transition = -0.0), NSSurfacelet(scales = 1, transition = 0.0)])) == 1
    @test t != NSSurfacelet(scales = 3, direction_levels = [1, 2, 2], tight = false)
    @test t != NSSurfacelet(scales = 3, direction_levels = [1, 2, 2], transition = 0.5)
    @test sprint(show, t) == "NSSurfacelet(scales = 3, direction_levels = [1, 2, 2], bernstein_order = 12, halfband_order = 1, transition = 1.0, tight = true)"
    layout = subband_layout(t, (16, 16, 16))
    @test nsubbands(t, (16, 16, 16)) == length(layout) == 1 + 6 + 12 + 12
    @test layout[1] == (scale = 0, direction = 0)
    @test layout[2] == (scale = 1, direction = 1)
    @test layout[8] == (scale = 2, direction = 1)
    @test supports_streaming(t) === true
    @test_throws "three axes" subband_layout(t, (16, 16))
    # Host arrays that are not `Array`s (a BitArray, a range) stay on the host.
    let t1 = NSSurfacelet(scales = 1), b = trues(16, 16, 16), r = reshape(1.0:4096.0, 16, 16, 16)
        @test analyze(t1, b) ≈ analyze(t1, Float64.(b))
        @test analyze(t1, r) ≈ analyze(t1, collect(r))
        s1 = Surfacelet(scales = 1, direction_levels = [1])
        r24 = reshape(1.0:(24^3), 24, 24, 24)
        @test Vector(analyze(s1, r24)) ≈ Vector(analyze(s1, collect(r24)))
    end
    let t1 = NSSurfacelet(scales = 1), z = complex.(randn(16, 16, 16), randn(16, 16, 16))
        @test_throws "coefficients of element type Float64, the workspace is for ComplexF64" analyze!(zeros(16, 16, 16, nsubbands(t1, (16, 16, 16))), t1, z)
    end
    @test_throws "scales must be ≥ 1" NSSurfacelet(scales = 0, direction_levels = Int[])
    @test_throws "length(direction_levels) must equal scales" NSSurfacelet(scales = 2, direction_levels = [1])
    @test_throws "direction_levels entries must be ≥ 0" NSSurfacelet(scales = 1, direction_levels = [-1])
    @test_throws "bernstein_order must be ≥ 1" NSSurfacelet(scales = 1, direction_levels = [1], bernstein_order = 0)
    @test_throws "halfband_order must be ≥ 0" NSSurfacelet(scales = 1, direction_levels = [1], halfband_order = -1)
    @test_throws "transition must be ≥ 0" NSSurfacelet(scales = 1, direction_levels = [1], transition = -1)
end

@testitem "ns_surfacelet — perfect reconstruction" tags = [:ns_surfacelet] begin
    using Random
    Random.seed!(90)
    for dims in ((16, 16, 16), (12, 20, 8))
        x = randn(dims...)
        for (J, L) in ((1, [0]), (2, [1, 2]), (3, [1, 1, 2]))
            t = NSSurfacelet(scales = J, direction_levels = L, bernstein_order = 6)
            c = analyze(t, x)
            n = nsubbands(t, dims)
            @test n == 1 + sum(l -> 3 * 2^l, L)
            @test all(i -> size(subband(t, c, i)) == dims, 1:n)
            @test maximum(abs, synthesize(t, c) .- x) < 1.0e-12
        end
    end
end

@testitem "ns_surfacelet — shift invariance" tags = [:ns_surfacelet] begin
    using Random
    Random.seed!(91)
    x = randn(16, 16, 16)
    t = NSSurfacelet(scales = 2, direction_levels = [1, 2], bernstein_order = 6)
    s = (3, -2, 5)
    a = analyze(t, x)
    b = analyze(t, circshift(x, s))
    for i in 1:nsubbands(t, size(x))
        @test maximum(abs, subband(t, b, i) .- circshift(subband(t, a, i), s)) < 1.0e-10
    end
end

@testitem "ns_surfacelet — a constant volume stays in the coarse residual" tags = [:ns_surfacelet] begin
    x = fill(1.75, 16, 16, 16)
    t = NSSurfacelet(scales = 2, direction_levels = [1, 1], bernstein_order = 6)
    c = analyze(t, x)
    @test maximum(abs, subband(t, c, 1) .- 1.75) < 1.0e-12
    for i in 2:nsubbands(t, size(x))
        @test maximum(abs, subband(t, c, i)) < 1.0e-12
    end
end

@testitem "ns_surfacelet — streaming analysis matches the materialized transform" tags = [:ns_surfacelet] begin
    using Random
    Random.seed!(92)
    x = randn(16, 16, 16)
    t = NSSurfacelet(scales = 2, direction_levels = [1, 2], bernstein_order = 6)
    ref = analyze(t, x)
    seen = Int[]
    r = analyze(t, x) do i, band
        push!(seen, i)
        @test maximum(abs, band .- subband(t, ref, i)) == 0.0
    end
    @test r === nothing
    # The directional subbands in order, then the lowpass.
    @test seen == [2:nsubbands(t, size(x)); 1]
    @test analyze((i, b) -> nothing, t, x; threaded = false) === nothing
end

@testitem "ns_surfacelet — streaming synthesis matches" tags = [:ns_surfacelet] begin
    using Random
    Random.seed!(93)
    x = randn(16, 16, 16)
    t = NSSurfacelet(scales = 2, direction_levels = [1, 2], bernstein_order = 6)
    c = analyze(t, x)
    r = similar(x)
    @test synthesize!(i -> subband(t, c, i), r, t) === r
    @test maximum(abs, r .- x) < 1.0e-12
    @test_throws DimensionMismatch synthesize!(i -> zeros(4, 4, 4), similar(x), t)
end

@testitem "ns_surfacelet — streaming complex and wrapped volumes" tags = [:ns_surfacelet] begin
    using Random
    rng = Xoshiro(96)
    t = NSSurfacelet(scales = 2, direction_levels = [1, 1], bernstein_order = 6)
    z = complex.(randn(rng, 16, 16, 16), randn(rng, 16, 16, 16))
    for tasks in (1, 2)
        ws = make_workspace(t, z; tasks)
        c = analyze(t, z; workspace = ws, threaded = false)
        streamed = similar(c)
        analyze((i, b) -> copyto!(subband(t, streamed, i), b), t, z; workspace = ws)
        @test streamed == c
        u, v = similar(z), similar(z)
        synthesize!(u, t, c; workspace = ws, threaded = false)
        @test synthesize!(i -> subband(t, c, i), v, t; workspace = ws) == u
        # The threaded forms still run after streaming.
        @test analyze(t, z; workspace = ws) == analyze(t, z; workspace = ws, threaded = false)
    end
    # A real volume and subbands the FFT plans cannot take directly.
    x = randn(rng, 18, 16, 16)
    xv = view(x, 2:17, :, :)
    ref = analyze(t, copy(xv))
    analyze(t, xv) do i, b
        @test b == subband(t, ref, i)
    end
    out = view(zeros(18, 16, 16), 2:17, :, :)
    synthesize!(i -> view(subband(t, ref, i), :, :, :), out, t)
    # Streaming adds the subbands on one task, as the unthreaded synthesis does;
    # the threaded synthesis adds per-task partial sums and rounds differently.
    @test out == synthesize(t, ref; threaded = false)
    @test out ≈ synthesize(t, ref) rtol = 1.0e-14
end

@testitem "ns_surfacelet — complex equals real + i·imag bit-for-bit" tags = [:ns_surfacelet] begin
    using Random
    Random.seed!(94)
    xr = randn(16, 16, 16); xi = randn(16, 16, 16)
    t = NSSurfacelet(scales = 2, direction_levels = [1, 1], bernstein_order = 6)
    cc = analyze(t, complex.(xr, xi))
    cr = analyze(t, xr)
    ci = analyze(t, xi)
    for i in 1:nsubbands(t, size(xr))
        @test maximum(abs, subband(t, cc, i) .- complex.(subband(t, cr, i), subband(t, ci, i))) == 0.0
    end
    @test maximum(abs, synthesize(t, cc) .- complex.(xr, xi)) < 1.0e-12
end

@testitem "ns_surfacelet — Float32" tags = [:ns_surfacelet] begin
    using Random
    Random.seed!(95)
    x = randn(Float32, 16, 16, 16)
    t = NSSurfacelet(scales = 2, direction_levels = [1, 1], bernstein_order = 6)
    c = analyze(t, x)
    @test eltype(c) == Float32
    @test eltype(subband(t, c, 1)) == Float32
    @test maximum(abs, synthesize(t, c) .- x) < 1.0f-4
end

@testitem "ns_surfacelet — workspace reuse is bit-identical and validated" tags = [:ns_surfacelet] begin
    using Random
    Random.seed!(96)
    x = randn(16, 16, 16)
    t = NSSurfacelet(scales = 2, direction_levels = [1, 2], bernstein_order = 6)
    ws = make_workspace(t, Float64, (16, 16, 16))
    @test transform(ws) === t && eltype(ws) == Float64 && size(ws) == (16, 16, 16)
    n = nsubbands(t, size(x))
    a = analyze(t, x)
    b = analyze(t, x; workspace = ws)
    c = analyze(ws, x)   # second call must not drift
    @test all(i -> subband(t, b, i) == subband(t, a, i), 1:n)
    @test all(i -> subband(t, c, i) == subband(t, a, i), 1:n)
    @test maximum(abs, synthesize(t, b; workspace = ws) .- x) < 1.0e-12
    @test maximum(abs, synthesize(ws, b) .- x) < 1.0e-12

    @test_throws DimensionMismatch analyze(t, randn(8, 8, 8); workspace = ws)
    @test_throws "element type" analyze(t, complex.(x, x); workspace = ws)
    wsj = make_workspace(NSSurfacelet(scales = 1, direction_levels = [1]), Float64, (16, 16, 16))
    @test_throws "the workspace was built for NSSurfacelet(scales = 1" analyze(t, x; workspace = wsj)
    # Per-scale directional depth and the `tight` normalization are baked into
    # the banks, so a workspace for another transform must be rejected rather
    # than silently overriding it.
    wsl = make_workspace(NSSurfacelet(scales = 2, direction_levels = [2, 2], bernstein_order = 6), Float64, (16, 16, 16))
    @test_throws ArgumentError analyze(t, x; workspace = wsl)
    wst = make_workspace(NSSurfacelet(scales = 2, direction_levels = [1, 2], bernstein_order = 6, tight = false), Float64, (16, 16, 16))
    @test_throws ArgumentError analyze(t, x; workspace = wst)
    @test_throws "three axes" make_workspace(t, Float64, (16, 16))
    @test_throws "tasks must be ≥ 1" make_workspace(t, Float64, (16, 16, 16); tasks = 0)
    # An integer element type means its float.
    @test eltype(make_workspace(t, Int, (16, 16, 16))) == Float64
end

@testitem "ns_surfacelet — fewer tasks give identical coefficients" tags = [:ns_surfacelet] begin
    using Random
    x = randn(Xoshiro(15), 16, 12, 20)
    t = NSSurfacelet(scales = 2, direction_levels = [1, 2], bernstein_order = 6)
    n = nsubbands(t, size(x))
    for Td in (Float64, ComplexF64)
        xd = Td.(x .+ (Td <: Complex ? 0.5im .* reverse(x; dims = 1) : 0))
        ref = analyze(t, xd; workspace = make_workspace(t, xd))
        for tasks in (1, 2)
            ws = make_workspace(t, xd; tasks)
            @test length(ws.engine.buffers) == min(tasks, Threads.nthreads())
            c = analyze(t, xd; workspace = ws)
            @test all(i -> subband(t, c, i) == subband(t, ref, i), 1:n)
            # The synthesis sums per-task partial sums, whose grouping follows
            # the task count.
            @test synthesize(t, c; workspace = ws) ≈ xd
        end
    end
end

@testitem "ns_surfacelet — a Parseval frame: the inverse is the adjoint" tags = [:ns_surfacelet] begin
    using Random, LinearAlgebra
    Random.seed!(98)
    # ‖Sx‖ = ‖x‖ and ⟨Sx, c⟩ = ⟨x, S⁻¹c⟩ for random coefficients c: the inverse is
    # the adjoint.  Odd and short axes included (a cardiac axis has 8 bins).
    for dims in ((16, 16, 16), (15, 20, 8)), (J, L) in ((1, [1]), (3, [1, 1, 2]))
        t = NSSurfacelet(scales = J, direction_levels = L, bernstein_order = 6)
        n = nsubbands(t, dims)
        inner(a, b) = sum(i -> dot(subband(t, a, i), subband(t, b, i)), 1:n)
        x = randn(dims...)
        c = analyze(t, x)
        @test inner(c, c) ≈ sum(abs2, x) rtol = 1.0e-12
        r = similar_coefficients(t, x)
        foreach(i -> subband(t, r, i) .= randn(dims...), 1:n)
        @test inner(c, r) ≈ dot(x, synthesize(t, r)) rtol = 1.0e-12
        z = complex.(x, randn(dims...))
        cz = analyze(t, z)
        @test real(inner(cz, cz)) ≈ sum(abs2, z) rtol = 1.0e-12
    end
end

@testitem "ns_surfacelet — a scale with an empty bandpass is refused" tags = [:ns_surfacelet] begin
    using Random
    # At 8 bins the third level's lowpass passes DC alone, which the fourth keeps
    # whole: scale 4 would be identically zero.
    t = NSSurfacelet(scales = 4, direction_levels = [0, 0, 0, 0], bernstein_order = 4)
    @test_throws "too small for 4 scales" make_workspace(t, Float64, (8, 8, 8))
    # A short axis alone is not refused while the others still split the band.
    x = randn(Xoshiro(3), 32, 32, 8)
    q = NSSurfacelet(scales = 3, direction_levels = [0, 1, 1], bernstein_order = 4)
    c = analyze(q, x)
    layout = subband_layout(q, size(x))
    @test all(j -> maximum(abs, subband(q, c, findfirst(==((scale = j, direction = 1)), layout))) > 0, 1:3)
    @test maximum(abs, synthesize(q, c) .- x) < 1.0e-12
end

@testitem "ns_surfacelet — uncached bandpass matches the cache" tags = [:ns_surfacelet] begin
    using Random
    for (Td, tight) in ((Float64, true), (Float64, false), (ComplexF64, true))
        x = randn(Xoshiro(7), Td, 16, 12, 20)
        t = NSSurfacelet(scales = 2, direction_levels = [1, 2], bernstein_order = 6, tight = tight)
        n = nsubbands(t, size(x))
        ws = make_workspace(t, x)
        wu = make_workspace(t, x; cache_bandpass = false)
        @test length(ws.band) == 2 && isempty(wu.band)
        c = analyze(t, x; workspace = ws)
        d = analyze(t, x; workspace = wu)
        @test all(i -> subband(t, d, i) ≈ subband(t, c, i), 1:n)
        @test maximum(abs, synthesize(t, d; workspace = wu) .- x) < 1.0e-12
        for threaded in (true, false)
            @test synthesize(t, d; workspace = wu, threaded) ≈ synthesize(t, c; workspace = ws, threaded)
        end
    end
end

@testitem "ns_surfacelet — adapting a workspace to the host keeps it" tags = [:ns_surfacelet] begin
    using Adapt, Random
    x = randn(Xoshiro(11), 16, 16, 16)
    t = NSSurfacelet(scales = 2, direction_levels = [1, 1], bernstein_order = 6)
    n = nsubbands(t, size(x))
    for cache in (true, false), tasks in (nothing, 1)
        ws = make_workspace(t, x; cache_bandpass = cache, tasks)
        wh = Adapt.adapt(Array, ws)
        @test transform(wh) == t
        @test wh.tasks === ws.tasks
        @test length(wh.engine.buffers) == length(ws.engine.buffers)
        @test length(wh.band) == length(ws.band)
        a, b = analyze(t, x; workspace = wh), analyze(t, x; workspace = ws)
        @test all(i -> subband(t, a, i) ≈ subband(t, b, i), 1:n)
    end
    # A view of a host array keeps the workspace on the host.
    @test make_workspace(t, view(x, :, :, :)) isa NSSurfaceletWorkspace{Float64, Float64, Array{Float64, 3}}
end

@testitem "ns_surfacelet — arrays the FFT plans cannot use directly" tags = [:ns_surfacelet] begin
    using Random
    # Real subbands and outputs that are not plain arrays of the plans' type go
    # through the workspace's one-task path; an integer volume is copied into a
    # real buffer first.  Both give the threaded path's result exactly.
    x = randn(Xoshiro(13), 12, 16, 10)
    t = NSSurfacelet(scales = 2, direction_levels = [1, 2], bernstein_order = 6)
    n = nsubbands(t, size(x))
    ws = make_workspace(t, x)
    c = analyze(t, x; workspace = ws)
    cv = view(similar(c), :, :, :, :)
    analyze!(cv, t, x; workspace = ws)
    @test all(i -> subband(t, cv, i) == subband(t, c, i), 1:n)
    y = synthesize(t, c; workspace = ws)
    @test synthesize(t, cv; workspace = ws) ≈ y
    yv = view(similar(x), :, :, :)
    @test synthesize!(yv, t, c; workspace = ws) == y
    @test maximum(abs, y .- x) < 1.0e-12
    xi = rand(Xoshiro(14), -3:3, size(x)...)
    ci = analyze(t, xi; workspace = ws)
    cf = analyze(t, float.(xi); workspace = ws)
    @test all(i -> subband(t, ci, i) == subband(t, cf, i), 1:n)
end

@testitem "NSSurfacelet — scales and direction_levels default from each other" tags = [:ns_surfacelet] begin
    @test NSSurfacelet(scales = 4).direction_levels == [1, 1, 2, 2]
    @test NSSurfacelet(scales = Int32(1)).direction_levels == [1]
    @test NSSurfacelet(direction_levels = [1, 2]).scales == 2
    @test NSSurfacelet(direction_levels = [1, 2]) == NSSurfacelet(scales = 2, direction_levels = [1, 2])
    @test_throws "length(direction_levels) must equal scales" NSSurfacelet(scales = 3, direction_levels = [1, 2])
    @test_throws "give `scales`, `direction_levels` or both" NSSurfacelet()
    @test_throws "scales must be ≥ 1" NSSurfacelet(scales = 0)
end

@testitem "NSSurfacelet — scales run from the coarsest to the finest" tags = [:ns_surfacelet] begin
    n = 16
    t = NSSurfacelet(direction_levels = [0, 1], bernstein_order = 6)
    layout = subband_layout(t, (n, n, n))
    @test layout[1].scale == 0
    @test [d.scale for d in layout[2:end]] == [fill(1, 3); fill(2, 6)]
    # A wave near the Nyquist frequency lands in the finest scale, scale 2, and
    # a slow one in the coarse lowpass or the coarsest scale.
    fast = [cospi(2 * (7 * (i - 1) + 6 * (j - 1)) / n) for i in 1:n, j in 1:n, _ in 1:n]
    slow = [cospi(2 * (i - 1) / n) for i in 1:n, _ in 1:n, _ in 1:n]
    energy(x) = (c = analyze(t, x); [sum(abs2, subband(t, c, i)) for i in eachindex(layout)])
    @test layout[argmax(energy(fast))].scale == 2
    @test layout[argmax(energy(slow))].scale in (0, 1)
end
