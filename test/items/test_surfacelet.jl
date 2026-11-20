@testitem "Surfacelet — construction, validation and equality" tags = [:surfacelet] begin
    using MultiscaleTransformsBase: supports_streaming
    t = Surfacelet(J = 2, L_array = [1, 2])
    @test t.J == 2
    @test t == Surfacelet(J = 2, L_array = Int8[1, 2])
    @test hash(t) == hash(Surfacelet(J = 2, L_array = [1, 2]))
    @test t != Surfacelet(J = 2, L_array = [1, 2], bo = 8)
    @test sprint(show, t) == "Surfacelet(J = 2, L_array = [1, 2], pyr_mode = 1.5, bo = 12, msize = 15, beta = 2.5, lambda = 4)"
    layout = subband_layout(t, (48, 48, 48))
    @test layout[1] == (scale = 3, hourglass = 0, wedge = 0)
    @test layout[2] == (scale = 1, hourglass = 1, wedge = 1)
    @test count(d -> d.scale == 1, layout) == 6
    @test count(d -> d.scale == 2, layout) == 12
    @test nsubbands(t, (48, 48, 48)) == 19
    @test supports_streaming(t) === false
    # A SurfBox level matrix: row K is the stage count per axis in hourglass K.
    q = Surfacelet(J = 1, L_array = [[-1 2 1; 1 -1 2; 2 1 -1]])
    @test q.levels[1] == ((0, 2, 1), (1, 0, 2), (2, 1, 0))
    @test nsubbands(q, (96, 96, 96)) == 25
    # The levels as given keep a concrete-enough element type: integers or matrices.
    @test eltype(t.L_array) == Union{Int, Matrix{Int}}
    @test t.L_array == [1, 2]
    @test q.L_array[1] isa Matrix{Int}
    mixed = Surfacelet(J = 2, L_array = Any[Int8(1), Int32[-1 2 1; 1 -1 2; 2 1 -1]])
    @test mixed.L_array[1] === 1
    @test mixed.L_array[2] isa Matrix{Int}
    @test mixed.levels == [t.levels[1], q.levels[1]]
    @test checkerboard_orders() == collect(4:2:18)
    @test_throws "J must be ≥ 1" Surfacelet(J = 0, L_array = Int[])
    @test_throws "length(L_array) must equal J" Surfacelet(J = 2, L_array = [1])
    @test_throws "directional levels must be ≥ 0" Surfacelet(J = 1, L_array = [-1])
    @test_throws "3 × 3" Surfacelet(J = 1, L_array = [ones(Int, 2, 2)])
    @test_throws "integer or a 3 × 3" Surfacelet(J = 1, L_array = [1.5])
    @test_throws "pyramid mode" Surfacelet(J = 1, L_array = [1], pyr_mode = 3)
    @test_throws ArgumentError Surfacelet(J = 1, L_array = [1], bo = 5)
    @test_throws "msize" Surfacelet(J = 1, L_array = [1], msize = 14)
    @test_throws "beta" Surfacelet(J = 1, L_array = [1], beta = 0)
    @test_throws "lambda" Surfacelet(J = 1, L_array = [1], lambda = 3)
end

@testitem "surfacelet — perfect reconstruction" tags = [:surfacelet] begin
    using Random
    rng = Xoshiro(100)
    cases = (
        ((48, 48, 48), 1.5, 2, [1, 2], 12),
        ((48, 24, 36), 1.5, 1, [2], 8),
        ((32, 32, 32), 1, 2, [1, 1], 4),
        ((64, 32, 32), 2, 2, [3, 1], 18),
        ((32, 32, 32), 2, 1, [[-1 2 0; 1 -1 1; 0 3 -1]], 6),
        ((24, 24, 24), 1.5, 1, [0], 12),
    )
    for (dims, mode, J, L, bo) in cases
        x = randn(rng, dims...)
        t = Surfacelet(J = J, L_array = L, pyr_mode = mode, bo = bo)
        ws = make_workspace(t, Float64, dims)
        c = analyze(t, x; workspace = ws)
        n = nsubbands(t, dims)
        @test sum(i -> length(subband(t, c, i)), 1:n) == round(Int, surfacelet_redundancy(t, dims) * prod(dims))
        @test maximum(abs, synthesize(t, c; workspace = ws) .- x) < 1.0e-11
    end
end

@testitem "surfacelet — threaded and serial runs agree" tags = [:surfacelet] begin
    using Random
    x = randn(Xoshiro(5), 48, 24, 36)
    t = Surfacelet(J = 2, L_array = [1, 2])
    n = nsubbands(t, size(x))
    ws = make_workspace(t, x)
    c = analyze(t, x; workspace = ws)
    # A workspace for one task plans its full-volume FFTs on one FFTW thread,
    # which may round differently.
    for (q, threaded) in ((ws, false), (make_workspace(t, x; tasks = 1), nothing))
        d = analyze(t, x; workspace = q, threaded)
        @test d ≈ c
        @test synthesize(t, d; workspace = q, threaded) ≈ synthesize(t, c; workspace = ws)
    end
    @test_throws "tasks must be ≥ 1" make_workspace(Surfacelet(J = 1, L_array = [1]), Float64, (24, 24, 24); tasks = 0)
end

@testitem "surfacelet — works without a caller-supplied workspace" tags = [:surfacelet] begin
    using Random
    x = randn(Xoshiro(101), 24, 24, 24)
    t = Surfacelet(J = 1, L_array = [1])
    c = analyze(t, x)
    @test c.layout.signal_size == size(x)
    @test maximum(abs, synthesize(t, c) .- x) < 1.0e-11
end

@testitem "surfacelet — complex equals real + i·imag bit-for-bit" tags = [:surfacelet] begin
    using Random
    rng = Xoshiro(102)
    for T in (Float64, Float32)
        xr = randn(rng, T, 48, 24, 36); xi = randn(rng, T, 48, 24, 36)
        z = complex.(xr, xi)
        t = Surfacelet(J = 2, L_array = [1, 2])
        n = nsubbands(t, size(z))
        cc = analyze(t, z)
        @test eltype(cc) == Complex{T}
        cr, ci = analyze(t, xr), analyze(t, xi)
        @test all(i -> subband(t, cc, i) == complex.(subband(t, cr, i), subband(t, ci, i)), 1:n)
        ws = make_workspace(t, z)
        @test eltype(ws) == Complex{T}
        # The workspace's FFTW.MEASURE plans may round differently from the
        # FFTW.ESTIMATE plans of a call without one.
        @test all(i -> subband(t, analyze(ws, z), i) ≈ subband(t, cc, i), 1:n)
        @test maximum(abs, synthesize(t, cc; workspace = ws) .- z) < (T == Float64 ? 1.0e-11 : 1.0f-4)
        @test_throws "element type" analyze(t, xr; workspace = ws)
    end
end

@testitem "surfacelet — critically sampled inside each hourglass" tags = [:surfacelet] begin
    # Every hourglass channel of a scale stores exactly as many coefficients as
    # the scale has voxels, however many wedges it is split into: the IRC
    # stages decimate by 2 per split.  So redundancy is 3 per scale times the
    # pyramid's geometric sum, and does not depend on the directional levels.
    dims = (96, 96, 96)
    red(L) = surfacelet_redundancy(Surfacelet(J = 2, L_array = L, bo = 4), dims)
    @test red([1, 1]) == red([3, 2]) == red([4, 4])
    # Mode 1.5: scale 1 at 96³, scale 2 at 64³, coarse at 32³.
    @test red([1, 1]) ≈ 3 + 3 * (64 / 96)^3 + (32 / 96)^3
    t = Surfacelet(J = 1, L_array = [2])
    c = analyze(t, randn(dims...))
    for K in 1:3
        bands = [subband(t, c, 1 + i) for i in (4K - 3):(4K)]
        @test sum(length, bands) == prod(dims)
        # Hourglass K keeps axis K at full length and decimates the other two.
        @test all(b -> size(b, K) == 96, bands)
        @test all(d -> d.hourglass == K, subband_layout(t, dims)[1 .+ ((4K - 3):(4K))])
    end
end

@testitem "surfacelet — a plane wave lands in its own hourglass" tags = [:surfacelet] begin
    n = 48
    t = Surfacelet(J = 1, L_array = [2])
    ws = make_workspace(t, Float64, (n, n, n))
    for (k, K) in (((14, 3, 2), 1), ((2, 15, 4), 2), ((3, -2, 13), 3))
        v = [
            cospi(2 * (k[1] * (I[1] - 1) + k[2] * (I[2] - 1) + k[3] * (I[3] - 1)) / n)
                for I in CartesianIndices((n, n, n))
        ]
        c = analyze(t, v; workspace = ws)
        e = [sum(abs2, subband(t, c, i)) for i in 2:13]
        hourglass = (argmax(e) - 1) ÷ 4 + 1
        @test hourglass == K
        # Directional: one wedge holds most of the energy of its hourglass.
        @test maximum(e) / sum(e[(4K - 3):(4K)]) > 0.6
    end
end

@testitem "surfacelet — a constant volume lands in the coarse subband" tags = [:surfacelet] begin
    dims = (48, 48, 48)
    x = fill(0.5, dims)
    t = Surfacelet(J = 2, L_array = [1, 1])
    ws = make_workspace(t, Float64, dims)
    c = analyze(t, x; workspace = ws)
    for i in 2:nsubbands(t, dims)
        @test maximum(abs, subband(t, c, i)) < 1.0e-12
    end
    @test maximum(abs, synthesize(t, c; workspace = ws) .- 0.5) < 1.0e-11
end

@testitem "surfacelet — Float32" tags = [:surfacelet] begin
    using Random
    dims = (24, 24, 24)
    x = randn(Xoshiro(103), Float32, dims...)
    t = Surfacelet(J = 1, L_array = [1])
    ws = make_workspace(t, Float32, dims)
    c = analyze(t, x; workspace = ws)
    @test eltype(c) == Float32
    @test maximum(abs, synthesize(t, c; workspace = ws) .- x) < 1.0f-4
end

@testitem "surfacelet — validation" tags = [:surfacelet] begin
    using Random
    dims = (24, 24, 24)
    t = Surfacelet(J = 1, L_array = [1])
    ws = make_workspace(t, Float64, dims)
    @test transform(ws) === t && size(ws) == dims && eltype(ws) == Float64
    # 1.5 does not divide 20, and 2^4 does not divide 24.
    @test_throws ArgumentError make_workspace(t, Float64, (20, 24, 24))
    @test_throws ArgumentError make_workspace(Surfacelet(J = 1, L_array = [7]), Float64, dims)
    @test_throws ArgumentError subband_layout(Surfacelet(J = 1, L_array = [7]), dims)
    @test_throws "three axes" make_workspace(t, Float64, (24, 24))
    @test_throws DimensionMismatch analyze(t, randn(48, 48, 48); workspace = ws)
    @test_throws "the workspace was built for Surfacelet(J = 1" analyze(Surfacelet(J = 1, L_array = [2]), randn(dims...); workspace = ws)
    c = analyze(t, randn(Xoshiro(104), dims...); workspace = ws)
    @test_throws "subband layout" synthesize(t, similar_coefficients(Surfacelet(J = 1, L_array = [2]), zeros(dims)); workspace = ws)
    @test_throws "subband layout" synthesize(t, similar_coefficients(t, zeros(48, 48, 48)); workspace = ws)
    @test_throws DimensionMismatch synthesize(Surfacelet(J = 1, L_array = [2]), c)
end

@testitem "surfacelet — serial calls use single-threaded FFT plans" tags = [:surfacelet] begin
    using Adapt, Random
    x = randn(Xoshiro(13), 48, 24, 36)
    t = Surfacelet(J = 2, L_array = [1, 2])
    ws = make_workspace(t, x; tasks = 2)
    for f in (ws.pfwd, ws.pbwd, ws.cfwd, ws.cbwd)
        @test f[1] !== f[2]
    end
    one = make_workspace(t, x; tasks = 1)
    @test one.pfwd[1] === one.pfwd[2]
    c = analyze(t, x; workspace = ws)
    d = analyze(t, x; workspace = ws, threaded = false)
    @test d ≈ c
    @test synthesize(t, d; workspace = ws, threaded = false) ≈ x
    # Adapting to the host keeps the task count and the transform.
    wh = Adapt.adapt(Array, ws)
    @test wh.tasks == 2 && wh.pfwd[1] !== wh.pfwd[2]
    @test transform(wh) == t
    @test synthesize(wh, analyze(wh, x)) ≈ x
    wz = Adapt.adapt(Array, make_workspace(t, ComplexF64, size(x)))
    @test eltype(wz) == ComplexF64
end
