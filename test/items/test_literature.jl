# Properties stated on the theory page and in docstrings, pinned against the
# numbers they quote, so that a change to the construction shows up as a
# deliberate edit to the documentation rather than a silent drift.

@testitem "ndfb — wedge-centre energy follows the transition width" tags = [:directional] begin
    # `transition` is the half-width of each wedge-boundary step, so at the
    # default of 1 the steps on either side of a wedge overlap at its centre and
    # no wedge has a flat passband. The `make_ndfb` docstring quotes the 71 %.
    n = 32
    # Hourglass 1 (ω₁ dominant), both transverse ratios 1/2: the centre of wedge
    # (p, q) = (2, 2) at `levels = 2`.
    k = (8, 4, 4)
    v = [
        cospi(2 * (k[1] * (I[1] - 1) + k[2] * (I[2] - 1) + k[3] * (I[3] - 1)) / n)
            for I in CartesianIndices((n, n, n))
    ]
    function share(transition)
        fb = make_ndfb((n, n, n), Float64; levels = 2, transition = transition)
        e = zeros(ndfb_nsubbands(fb))
        ndfb_decompose(v, fb) do idx, band
            e[idx] = sum(abs2, band)
            return nothing
        end
        @test argmax(e) == ndfb_subband_index(fb, 1, 2, 2)
        return maximum(e) / sum(e)
    end
    @test isapprox(share(1.0), 0.71; atol = 0.01)
    # What the wedge steps leave at 0.5 and below is hourglass overlap, which
    # `transition` does not control.
    @test share(0.5) > 0.99
    @test isapprox(share(0.0), share(0.5); atol = 1.0e-3)
end

@testitem "ndfb — subband count is 3·2^L, the paper's NDFB count" tags = [:directional] begin
    # Lu & Do (2007), Theorem 1: an L-level NDFB in 3-D has 3·2^L subbands,
    # split between the two transverse planes of each hourglass.
    for L in 0:5
        fb = make_ndfb((8, 8, 8), Float64; levels = L, order = 4)
        @test ndfb_nsubbands(fb) == 3 * 2^L
        @test sum(fb.levels) == L
        @test fb.levels[1] - fb.levels[2] in (0, 1)
    end
end

@testitem "surfacelet — redundancy matches the paper's figures" tags = [:surfacelet] begin
    # Lu & Do (2007), Section VI-A: the pyramid modes give redundancies of about
    # 6.4, 4.0 and 3.4 in 3-D (SurfBox `Surfdec.m` documents the same figures),
    # whatever the number of directions.
    dims = (192, 192, 192)
    for (mode, r) in ((1, 6.4), (1.5, 4.0), (2, 3.4))
        for L in ([1, 1, 1], [3, 3, 3])
            p = SurfaceletParams(J = 3, L_array = L, pyr_mode = mode, bo = 4)
            @test isapprox(surfacelet_redundancy(make_surfacelet(p, dims)), r; atol = 0.1)
        end
    end
end

@testitem "positional-size convenience constructors" tags = [:primitives] begin
    @test size(make_conv3d_fft_plan(8, 10, 12).ker) == size(make_conv3d_fft_plan((8, 10, 12)).ker)
    @test make_hourglass(8, 10, 12; order = 4).dims == (8, 10, 12)
    @test ndfb_nsubbands(make_ndfb(8, 10, 12; levels = 1, order = 4)) == 6

    p = SurfaceletParams(J = 1, L_array = [1])
    @test surfacelet_redundancy(make_surfacelet(p, 24, 24, 24)) ==
        surfacelet_redundancy(make_surfacelet(p, (24, 24, 24)))

    np = NSSurfaceletParams(J = 1, L_array = [1], order = 4)
    ws = make_ns_surfacelet_workspace(np, 16, 16, 16)
    x = randn(16, 16, 16)
    @test ns_surfacelet_inverse(ns_surfacelet_forward(x, np; workspace = ws), np; workspace = ws) ≈ x
end

@testitem "conv3d_sep_fft — allocating form and single-threaded plan" tags = [:primitives] begin
    using Random
    x = randn(Xoshiro(3), 16, 16, 16)
    h = randn(Xoshiro(4), 7)
    plan = make_conv3d_fft_plan(size(x); threading = Disabled())
    @test conv3d_sep_fft(x, h, h, h) ≈ conv3d_sep_fft!(similar(x), x, h, h, h, plan)
    @test conv3d_sep_fft(x, h, h, h) ≈ conv3d_sep(x, h, h, h; boundary = :periodic)
end

@testitem "Bernstein helpers — default precision and the 1-D kernel" tags = [:filters] begin
    @test Surfacelets.dft_frequencies(8) == Surfacelets.dft_frequencies(Float64, 8)
    @test length(Surfacelets.dft_frequencies(8; half = true)) == 5
    # A 1-D mapping kernel is a single contraction of the Bernstein table.
    ω = Surfacelets.dft_frequencies(16)
    tbl = Surfacelets.bernstein_table(4, ω)
    c = [1.0, 0.5, 0.0, -0.5, -1.0]
    @test Surfacelets.mapping_response(c, (tbl,)) ≈ vec(c' * tbl)
end

@testitem "ndfb_decompose! — wrong number of destination subbands" tags = [:directional] begin
    fb = make_ndfb((8, 8, 8), Float64; levels = 1, order = 4)
    x = randn(8, 8, 8)
    @test_throws DimensionMismatch ndfb_decompose!([similar(x)], x, fb)
end
