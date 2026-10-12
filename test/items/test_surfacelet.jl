@testitem "SurfaceletParams — construction and validation" tags = [:surfacelet] begin
    p = SurfaceletParams(J = 2, L_array = [1, 2])
    @test p.J == 2
    @test eltype(p) == Float64
    @test surfacelet_nsubbands(p, 1) == 6
    @test surfacelet_nsubbands(p, 2) == 12
    @test occursin("SurfaceletParams", sprint(show, p))
    # A SurfBox level matrix: row K is the stage count per axis in hourglass K.
    q = SurfaceletParams(J = 1, L_array = [[-1 2 1; 1 -1 2; 2 1 -1]])
    @test q.levels[1] == ((0, 2, 1), (1, 0, 2), (2, 1, 0))
    @test surfacelet_nsubbands(q, 1) == 24
    @test checkerboard_orders() == collect(4:2:18)
    @test_throws ArgumentError SurfaceletParams(J = 0, L_array = Int[])
    @test_throws ArgumentError SurfaceletParams(J = 2, L_array = [1])
    @test_throws ArgumentError SurfaceletParams(J = 1, L_array = [-1])
    @test_throws ArgumentError SurfaceletParams(J = 1, L_array = [ones(Int, 2, 2)])
    @test_throws ArgumentError SurfaceletParams(J = 1, L_array = [1], pyr_mode = 3)
    @test_throws ArgumentError SurfaceletParams(J = 1, L_array = [1], bo = 5)
    @test_throws ArgumentError SurfaceletParams(J = 1, L_array = [1], msize = 14)
    @test_throws ArgumentError SurfaceletParams(J = 1, L_array = [1], beta = 0)
    @test_throws ArgumentError SurfaceletParams(J = 1, L_array = [1], lambda = 3)
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
        p = SurfaceletParams(J = J, L_array = L, pyr_mode = mode, bo = bo)
        pl = make_surfacelet(p, dims, Float64)
        c = surfacelet_forward(x, p; plan = pl)
        @test length(c.subbands) == J
        @test all(j -> length(c.subbands[j]) == surfacelet_nsubbands(p, j), 1:J)
        @test maximum(abs, surfacelet_inverse(c, p; plan = pl) .- x) < 1.0e-11
    end
end

@testitem "surfacelet — works without a caller-supplied plan" tags = [:surfacelet] begin
    using Random
    x = randn(Xoshiro(101), 24, 24, 24)
    p = SurfaceletParams(J = 1, L_array = [1])
    c = surfacelet_forward(x, p)
    @test c.dims == size(x)
    @test maximum(abs, surfacelet_inverse(c, p) .- x) < 1.0e-11
end

@testitem "surfacelet — critically sampled inside each hourglass" tags = [:surfacelet] begin
    # Every hourglass channel of a scale stores exactly as many coefficients as
    # the scale has voxels, however many wedges it is split into: the IRC
    # stages decimate by 2 per split.  So redundancy is 3 per scale times the
    # pyramid's geometric sum, and does not depend on the directional levels.
    dims = (96, 96, 96)
    red(L) = surfacelet_redundancy(make_surfacelet(SurfaceletParams(J = 2, L_array = L, bo = 4), dims))
    @test red([1, 1]) == red([3, 2]) == red([4, 4])
    # Mode 1.5: scale 1 at 96³, scale 2 at 64³, coarse at 32³.
    @test red([1, 1]) ≈ 3 + 3 * (64 / 96)^3 + (32 / 96)^3
    pl = make_surfacelet(SurfaceletParams(J = 1, L_array = [2]), dims)
    c = surfacelet_forward(randn(dims...), SurfaceletParams(J = 1, L_array = [2]); plan = pl)
    for K in 1:3
        bands = c.subbands[1][(4K - 3):(4K)]
        @test sum(length, bands) == prod(dims)
        # Hourglass K keeps axis K at full length and decimates the other two.
        @test all(b -> size(b, K) == 96, bands)
    end
end

@testitem "surfacelet — a plane wave lands in its own hourglass" tags = [:surfacelet] begin
    n = 48
    p = SurfaceletParams(J = 1, L_array = [2])
    pl = make_surfacelet(p, (n, n, n))
    for (k, K) in (((14, 3, 2), 1), ((2, 15, 4), 2), ((3, -2, 13), 3))
        v = [
            cospi(2 * (k[1] * (I[1] - 1) + k[2] * (I[2] - 1) + k[3] * (I[3] - 1)) / n)
                for I in CartesianIndices((n, n, n))
        ]
        c = surfacelet_forward(v, p; plan = pl)
        e = [sum(abs2, b) for b in c.subbands[1]]
        hourglass = (argmax(e) - 1) ÷ 4 + 1
        @test hourglass == K
        # Directional: one wedge holds most of the energy of its hourglass.
        @test maximum(e) / sum(e[(4K - 3):(4K)]) > 0.6
    end
end

@testitem "surfacelet — a constant volume lands in the coarse subband" tags = [:surfacelet] begin
    dims = (48, 48, 48)
    x = fill(0.5, dims)
    p = SurfaceletParams(J = 2, L_array = [1, 1])
    pl = make_surfacelet(p, dims, Float64)
    c = surfacelet_forward(x, p; plan = pl)
    for scale in c.subbands, band in scale
        @test maximum(abs, band) < 1.0e-12
    end
    @test maximum(abs, surfacelet_inverse(c, p; plan = pl) .- 0.5) < 1.0e-11
end

@testitem "surfacelet — Float32" tags = [:surfacelet] begin
    using Random
    dims = (24, 24, 24)
    x = randn(Xoshiro(103), Float32, dims...)
    p = SurfaceletParams(J = 1, L_array = [1], precision = Float32)
    pl = make_surfacelet(p, dims, Float32)
    c = surfacelet_forward(x, p; plan = pl)
    @test eltype(c) == Float32
    @test maximum(abs, surfacelet_inverse(c, p; plan = pl) .- x) < 1.0f-4
end

@testitem "surfacelet — validation" tags = [:surfacelet] begin
    using Random
    dims = (24, 24, 24)
    p = SurfaceletParams(J = 1, L_array = [1])
    pl = make_surfacelet(p, dims, Float64)
    # 1.5 does not divide 20, and 2^4 does not divide 24.
    @test_throws ArgumentError make_surfacelet(p, (20, 24, 24), Float64)
    @test_throws ArgumentError make_surfacelet(SurfaceletParams(J = 1, L_array = [7]), dims)
    @test_throws DimensionMismatch surfacelet_forward(randn(48, 48, 48), p; plan = pl)
    c = surfacelet_forward(randn(Xoshiro(104), dims...), p; plan = pl)
    @test occursin("SurfaceletCoefficients", sprint(show, c))
    bad = SurfaceletCoefficients(c.dims, c.coarse, [c.subbands[1][1:2]])
    @test_throws DimensionMismatch surfacelet_inverse(bad, p; plan = pl)
    # Subbands of hourglass 2 in hourglass 1's slots have the wrong shape.
    swapped = SurfaceletCoefficients(c.dims, c.coarse, [[c.subbands[1][3:4]; c.subbands[1][1:2]; c.subbands[1][5:6]]])
    @test_throws DimensionMismatch surfacelet_inverse(swapped, p; plan = pl)
    @test_throws DimensionMismatch surfacelet_inverse(
        SurfaceletCoefficients(c.dims, zeros(4, 4, 4), c.subbands), p; plan = pl
    )
    p2 = SurfaceletParams(J = 2, L_array = [1, 1])
    @test_throws ArgumentError surfacelet_inverse(c, p2)
end
