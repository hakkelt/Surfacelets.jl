@testitem "bernstein_table — partition of unity and endpoints" tags = [:filters] begin
    w = range(0, 2π; length = 33)
    for N in (1, 4, 12)
        tbl = Surfacelets.bernstein_table(N, collect(w))
        @test size(tbl) == (N + 1, length(w))
        # Bernstein basis functions sum to 1 for every x.
        @test maximum(abs, vec(sum(tbl; dims = 1)) .- 1) < 1.0e-12
        @test all(>=(-1.0e-15), tbl)
    end
    # ω = 0 gives x = 0: all the mass sits on the j = 0 basis function.
    t0 = Surfacelets.bernstein_table(6, [0.0])
    @test t0[1, 1] ≈ 1.0
    @test maximum(abs, t0[2:end, 1]) < 1.0e-14
    # ω = π gives x = 1: all the mass sits on j = N.
    tp = Surfacelets.bernstein_table(6, [Float64(π)])
    @test tp[end, 1] ≈ 1.0
    @test_throws ArgumentError Surfacelets.bernstein_table(0, [0.0])
end

@testitem "mapping_coefficients — sign pattern of the passband support" tags = [:filters] begin
    c = Surfacelets.mapping_coefficients(4, 2, (1,))
    # S = {x₁ ≥ x₂}: +1 below the diagonal in index space, -1 above, 0 on it.
    for j1 in 0:4, j2 in 0:4
        expected = j1 > j2 ? 1 : (j1 < j2 ? -1 : 0)
        @test c[j1 + 1, j2 + 1] == expected
    end
    c3 = Surfacelets.mapping_coefficients(3, 3, (1,))
    @test c3[4, 1, 1] == 1      # x₁ largest
    @test c3[1, 4, 1] == -1     # x₂ largest
    @test c3[2, 2, 2] == 0      # tie
    @test_throws ArgumentError Surfacelets.mapping_coefficients(3, 3, (1, 2, 3))
    @test_throws ArgumentError Surfacelets.mapping_coefficients(3, 3, (4,))
end

@testitem "halfband_split — shape and complementarity" tags = [:filters] begin
    for m in 0:3
        @test Surfacelets.halfband_split(1.0, m) ≈ 1.0
        @test Surfacelets.halfband_split(-1.0, m) ≈ 0.0
        @test Surfacelets.halfband_split(0.0, m) ≈ 0.5
        # f₀(x) + f₀(-x) = 1 is what makes the filter bank a partition of unity.
        for x in range(-1, 1; length = 21)
            f = Surfacelets.halfband_split(x, m)
            @test abs(f + Surfacelets.halfband_split(-x, m) - 1) < 1.0e-12
            @test -1.0e-12 <= f <= 1 + 1.0e-12
        end
    end
    @test_throws ArgumentError Surfacelets.halfband_split(0.0, -1)
end

@testitem "hourglass — weights are a partition of unity in [0,1]" tags = [:directional] begin
    fb = make_hourglass((12, 14, 10), Float64; order = 6, poly = 1)
    w = ntuple(i -> hourglass_weight(fb, i), 3)
    @test maximum(abs, w[1] .+ w[2] .+ w[3] .- 1) < 1.0e-14
    for wi in w
        @test minimum(wi) >= -1.0e-14
        @test maximum(wi) <= 1 + 1.0e-14
    end
    @test_throws ArgumentError hourglass_mask(fb, 4)
    @test_throws ArgumentError hourglass_weight(fb, 4)
end

@testitem "hourglass — tight masks are a Parseval frame" tags = [:directional] begin
    using Random
    fb = make_hourglass((12, 14, 10), Float64; order = 6, poly = 1)
    @test fb.tight
    m = ntuple(i -> hourglass_mask(fb, i), 3)
    # sum of squares, not sum, is what is 1 for the tight normalisation
    @test maximum(abs, m[1] .^ 2 .+ m[2] .^ 2 .+ m[3] .^ 2 .- 1) < 1.0e-14
    # ... and it shows up as energy preservation of the analysis operator.
    x = randn(Xoshiro(71), 12, 14, 10)
    s = hourglass_decompose(x, fb)
    @test abs(sum(sum(abs2, b) for b in s) / sum(abs2, x) - 1) < 1.0e-12

    # The untight bank is the partition of unity, and is not energy preserving.
    loose = make_hourglass((12, 14, 10), Float64; order = 6, poly = 1, tight = false)
    @test !loose.tight
    ml = ntuple(i -> hourglass_mask(loose, i), 3)
    @test maximum(abs, ml[1] .+ ml[2] .+ ml[3] .- 1) < 1.0e-14
    sl = hourglass_decompose(x, loose)
    @test sum(sum(abs2, b) for b in sl) / sum(abs2, x) < 0.9
    # Both invert exactly.
    @test maximum(abs, hourglass_reconstruct(s, fb) .- x) < 1.0e-12
    @test maximum(abs, hourglass_reconstruct(sl, loose) .- x) < 1.0e-12
end

@testitem "hourglass — perfect reconstruction" tags = [:directional] begin
    using Random
    Random.seed!(70)
    for dims in ((16, 16, 16), (12, 20, 8))
        x = randn(dims...)
        for (order, poly) in ((4, 0), (8, 1), (12, 2))
            fb = make_hourglass(dims, Float64; order = order, poly = poly)
            s = hourglass_decompose(x, fb)
            @test all(b -> size(b) == dims, s)
            @test maximum(abs, hourglass_reconstruct(s, fb) .- x) < 1.0e-12
        end
    end
end

@testitem "hourglass — axis-aligned plane waves land in the matching subband" tags = [:directional] begin
    n = 16
    fb = make_hourglass((n, n, n), Float64; order = 8, poly = 1)
    for ax in 1:3
        v = Array{Float64, 3}(undef, n, n, n)
        for I in CartesianIndices(v)
            t = (Tuple(I) .- 1)[ax]
            v[I] = cospi(2 * 6 * t / n)
        end
        e = [sum(abs2, b) for b in hourglass_decompose(v, fb)]
        e ./= sum(e)
        @test e[ax] > 0.999
        @test sum(e[setdiff(1:3, ax)]) < 1.0e-3
    end
end

@testitem "hourglass — a diagonal wave splits between its two axes" tags = [:directional] begin
    # Equal frequency along axes 1 and 2 sits exactly on the P₁/P₂ boundary, so
    # it must divide evenly between those subbands and leave P₃ empty.
    n = 16
    fb = make_hourglass((n, n, n), Float64; order = 8, poly = 1)
    v = Array{Float64, 3}(undef, n, n, n)
    for I in CartesianIndices(v)
        i, j, _ = Tuple(I) .- 1
        v[I] = cospi(2 * 5 * (i + j) / n)
    end
    e = [sum(abs2, b) for b in hourglass_decompose(v, fb)]
    e ./= sum(e)
    @test abs(e[1] - e[2]) < 1.0e-6
    @test e[3] < 1.0e-6
end

@testitem "hourglass — sharper with a higher Bernstein order" tags = [:directional] begin
    # Mean absolute deviation of mask 1 from the ideal indicator must fall as the
    # Bernstein order grows (Lu & Do 2008, Fig. 10).
    n = 24
    function mae(order)
        fb = make_hourglass((n, n, n), Float64; order = order, poly = 1)
        m = hourglass_mask(fb, 1)
        fold(k) = let w = (k - 1) * 2π / n
            min(w, 2π - w)
        end
        total = 0.0
        cnt = 0
        for k in axes(m, 3), j in axes(m, 2), p in axes(m, 1)
            w1, w2, w3 = fold(p), fold(j), fold(k)
            mx = max(w2, w3)
            abs(w1 - mx) > 1.0e-9 || continue
            total += abs(m[p, j, k] - (w1 > mx ? 1.0 : 0.0))
            cnt += 1
        end
        return total / cnt
    end
    e4, e12, e20 = mae(4), mae(12), mae(20)
    @test e12 < e4
    @test e20 < e12
end

@testitem "hourglass — exact circular-shift equivariance" tags = [:directional] begin
    using Random
    Random.seed!(71)
    n = 16
    x = randn(n, n, n)
    fb = make_hourglass((n, n, n), Float64; order = 6, poly = 1)
    s = (3, -5, 2)
    a = hourglass_decompose(x, fb)
    b = hourglass_decompose(circshift(x, s), fb)
    for i in 1:3
        @test maximum(abs, b[i] .- circshift(a[i], s)) < 1.0e-10
    end
end

@testitem "hourglass — complex equals real + i·imag bit-for-bit" tags = [:directional] begin
    using Random
    Random.seed!(72)
    n = 16
    xr = randn(n, n, n); xi = randn(n, n, n)
    fb = make_hourglass((n, n, n), ComplexF64; order = 6, poly = 1)
    sc = hourglass_decompose(complex.(xr, xi), fb)
    sr = hourglass_decompose(xr, fb)
    si = hourglass_decompose(xi, fb)
    for i in 1:3
        @test maximum(abs, sc[i] .- complex.(sr[i], si[i])) == 0.0
    end
    @test maximum(abs, hourglass_reconstruct(sc, fb) .- complex.(xr, xi)) < 1.0e-12
end

@testitem "hourglass — Float32 plan" tags = [:directional] begin
    using Random
    Random.seed!(73)
    n = 16
    x = randn(Float32, n, n, n)
    fb = make_hourglass((n, n, n), Float32; order = 6, poly = 1)
    s = hourglass_decompose(x, fb)
    @test all(b -> eltype(b) == Float32, s)
    @test maximum(abs, hourglass_reconstruct(s, fb) .- x) < 1.0f-4
end

@testitem "hourglass — argument validation" tags = [:directional] begin
    @test_throws ArgumentError make_hourglass((0, 4, 4), Float64)
    @test_throws ArgumentError make_hourglass((8, 8, 8), Float64; order = 0)
    fb = make_hourglass((8, 8, 8), Float64; order = 4)
    @test_throws DimensionMismatch hourglass_decompose(randn(8, 8, 4), fb)
end
