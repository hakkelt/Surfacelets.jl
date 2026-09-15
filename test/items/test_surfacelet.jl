@testitem "SurfaceletParams — construction and validation" tags = [:surfacelet] begin
    p = SurfaceletParams(J = 2, L_array = [1, 2])
    @test p.J == 2
    @test eltype(p) == Float64
    @test surfacelet_nsubbands(p, 1) == 6
    @test surfacelet_nsubbands(p, 2) == 12
    @test occursin("SurfaceletParams", sprint(show, p))
    @test_throws ArgumentError SurfaceletParams(J = 0, L_array = Int[])
    @test_throws ArgumentError SurfaceletParams(J = 2, L_array = [1])
    @test_throws ArgumentError SurfaceletParams(J = 1, L_array = [-1])
    @test_throws ArgumentError SurfaceletParams(J = 1, L_array = [1], poly = -1)
    @test_throws ArgumentError SurfaceletParams(J = 1, L_array = [1], transition = -1)
    @test_throws ArgumentError SurfaceletParams(J = 1, L_array = [1], radial_transition = 0)
    @test_throws ArgumentError SurfaceletParams(J = 1, L_array = [1], angular_transition = 0)
end

@testitem "surfacelet — masks satisfy Σ V² = 1" tags = [:surfacelet] begin
    # This is the perfect-reconstruction condition itself: the transform stores
    # √W with ΣW = 1, so synthesis is the analysis adjoint.
    dims = (16, 20, 12)
    p = SurfaceletParams(J = 2, L_array = [1, 2])
    pl = make_surfacelet(p, dims, Float64)
    acc = zeros(dims)
    mask = Array{Float64, 3}(undef, dims)
    for j in 1:p.J, k in 1:surfacelet_nsubbands(p, j)
        Surfacelets._fill_surfacelet_mask!(mask, pl, j, k)
        acc .+= mask .^ 2
    end
    Surfacelets._fill_coarse_mask!(mask, pl)
    acc .+= mask .^ 2
    @test maximum(abs, acc .- 1) < 1.0e-12
end

@testitem "surfacelet — masks are even in ω" tags = [:surfacelet] begin
    # Evenness is what keeps each subband real.  It is not automatic: on the DFT
    # grid ω → -ω fixes the Nyquist bin of an even-length axis, so a
    # sign-sensitive quantity like ω₂/ω₁ can disagree between a bin and its
    # negation unless the mask is evaluated at a canonical representative.
    dims = (16, 16, 16)
    p = SurfaceletParams(J = 2, L_array = [2, 2])
    pl = make_surfacelet(p, dims, Float64)
    mask = Array{Float64, 3}(undef, dims)
    neg(k, n) = mod(-(k - 1), n) + 1
    for j in 1:p.J, k in 1:surfacelet_nsubbands(p, j)
        Surfacelets._fill_surfacelet_mask!(mask, pl, j, k)
        worst = 0.0
        for k3 in 1:dims[3], k2 in 1:dims[2], k1 in 1:dims[1]
            m = mask[neg(k1, dims[1]), neg(k2, dims[2]), neg(k3, dims[3])]
            worst = max(worst, abs(mask[k1, k2, k3] - m))
        end
        @test worst < 1.0e-14
    end
end

@testitem "surfacelet — perfect reconstruction" tags = [:surfacelet] begin
    using Random
    Random.seed!(100)
    for dims in ((16, 16, 16), (24, 16, 20))
        x = randn(dims...)
        for (J, L) in ((1, [0]), (1, [1]), (2, [1, 1]), (2, [2, 2]))
            p = SurfaceletParams(J = J, L_array = L)
            pl = make_surfacelet(p, dims, Float64)
            c = surfacelet_forward(x, p; plan = pl)
            @test length(c.subbands) == J
            @test all(j -> length(c.subbands[j]) == 3 * 2^L[j], 1:J)
            @test maximum(abs, surfacelet_inverse(c, p; plan = pl) .- x) < 1.0e-11
        end
    end
end

@testitem "surfacelet — works without a caller-supplied plan" tags = [:surfacelet] begin
    using Random
    Random.seed!(101)
    x = randn(16, 16, 16)
    p = SurfaceletParams(J = 1, L_array = [1])
    c = surfacelet_forward(x, p)
    @test c.dims == size(x)
    @test maximum(abs, surfacelet_inverse(c, p) .- x) < 1.0e-11
end

@testitem "surfacelet — subbands really are decimated" tags = [:surfacelet] begin
    dims = (32, 32, 32)
    p = SurfaceletParams(J = 3, L_array = [1, 1, 1])
    pl = make_surfacelet(p, dims, Float64)
    # Coarser scales cover smaller frequency shells, so their boxes shrink.
    fine = maximum(prod(b.size) for b in pl.boxes[1])
    mid = maximum(prod(b.size) for b in pl.boxes[2])
    coarse = maximum(prod(b.size) for b in pl.boxes[3])
    @test mid < fine
    @test coarse < mid
    @test prod(pl.coarse_box.size) < prod(dims)
    # And the whole thing is leaner than the nonsubsampled transform at the same
    # settings, which would be 1 + 3·2 + 3·2 + 3·2 = 19.
    @test surfacelet_redundancy(pl) < 19
end

@testitem "surfacelet — energy is tight in the weighted sense" tags = [:surfacelet] begin
    using Random
    Random.seed!(102)
    dims = (16, 16, 16)
    x = randn(dims...)
    p = SurfaceletParams(J = 2, L_array = [1, 1])
    pl = make_surfacelet(p, dims, Float64)
    c = surfacelet_forward(x, p; plan = pl)
    # sum_i N_i * norm(s_i)^2 == N * norm(x)^2, from Parseval on each box.
    weighted(b) = length(b) * sum(abs2, b)
    total = weighted(c.coarse) +
        sum(weighted(band) for scale in c.subbands for band in scale)
    @test abs(total / (prod(dims) * sum(abs2, x)) - 1) < 1.0e-10
end

@testitem "surfacelet — a constant volume lands in the coarse subband" tags = [:surfacelet] begin
    dims = (16, 16, 16)
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
    Random.seed!(103)
    dims = (16, 16, 16)
    x = randn(Float32, dims...)
    p = SurfaceletParams(J = 1, L_array = [1], precision = Float32)
    pl = make_surfacelet(p, dims, Float32)
    c = surfacelet_forward(x, p; plan = pl)
    @test eltype(c) == Float32
    @test maximum(abs, surfacelet_inverse(c, p; plan = pl) .- x) < 1.0f-4
end

@testitem "surfacelet — validation" tags = [:surfacelet] begin
    using Random
    Random.seed!(104)
    dims = (16, 16, 16)
    p = SurfaceletParams(J = 1, L_array = [1])
    pl = make_surfacelet(p, dims, Float64)
    @test_throws ArgumentError make_surfacelet(p, (0, 4, 4), Float64)
    @test_throws DimensionMismatch surfacelet_forward(randn(8, 8, 8), p; plan = pl)
    c = surfacelet_forward(randn(dims...), p; plan = pl)
    @test occursin("SurfaceletCoefficients", sprint(show, c))
    bad = SurfaceletCoefficients(c.dims, c.coarse, [c.subbands[1][1:2]])
    @test_throws DimensionMismatch surfacelet_inverse(bad, p; plan = pl)
    p2 = SurfaceletParams(J = 2, L_array = [1, 1])
    @test_throws ArgumentError surfacelet_inverse(c, p2)
end
