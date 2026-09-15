@testitem "conv3d_sep — separability against three 1-D passes" tags = [:primitives] begin
    using Random
    Random.seed!(40)
    x = randn(9, 11, 7)
    h1 = randn(5); h2 = randn(3); h3 = randn(7)

    # Reference: brute-force per-axis filtering with symmetric extension.
    fold(i, n) = begin
        1 <= i <= n && return i
        n == 1 && return 1
        p = 2 * (n - 1)
        m = mod(i - 1, p)
        return n - abs(m - (n - 1))
    end
    function filt(src, h, dim)
        n1, n2, n3 = size(src)
        out = similar(src)
        l = length(h); c = (l + 1) ÷ 2
        for k in 1:n3, j in 1:n2, i in 1:n1
            acc = 0.0
            for t in 1:l
                off = c - t
                ii = dim == 1 ? fold(i + off, n1) : i
                jj = dim == 2 ? fold(j + off, n2) : j
                kk = dim == 3 ? fold(k + off, n3) : k
                acc += h[t] * src[ii, jj, kk]
            end
            out[i, j, k] = acc
        end
        return out
    end
    ref = filt(filt(filt(x, h1, 1), h2, 2), h3, 3)
    @test maximum(abs, conv3d_sep(x, h1, h2, h3) .- ref) < 1.0e-12
end

@testitem "conv3d_sep — DC gain and constant preservation" tags = [:primitives] begin
    h = [0.25, 0.5, 0.25]
    x = fill(3.0, 8, 8, 8)
    for bnd in (:symmetric, :periodic)
        y = conv3d_sep(x, h, h, h; boundary = bnd)
        @test maximum(abs, y .- 3.0) < 1.0e-12
    end
end

@testitem "conv3d_sep — filter longer than volume (no interior slab)" tags = [:primitives] begin
    using Random
    Random.seed!(41)
    x = randn(3, 4, 3)
    h = randn(9)
    y = conv3d_sep(x, h, h, h; boundary = :symmetric)
    @test size(y) == size(x)
    @test all(isfinite, y)
end

@testitem "conv3d_sep — periodic boundary is shift-equivariant" tags = [:primitives] begin
    using Random
    Random.seed!(42)
    x = randn(16, 16, 16)
    h = randn(5)
    s = (3, -2, 5)
    y = conv3d_sep(x, h, h, h; boundary = :periodic)
    ys = conv3d_sep(circshift(x, s), h, h, h; boundary = :periodic)
    @test maximum(abs, ys .- circshift(y, s)) < 1.0e-12
end

@testitem "conv3d_sep — zero boundary matches padded reference" tags = [:primitives] begin
    using Random
    Random.seed!(43)
    x = randn(6, 5, 4)
    h = [1.0, 2.0, 3.0]
    n1, n2, n3 = size(x)
    xp = zeros(n1 + 4, n2 + 4, n3 + 4)
    xp[3:(n1 + 2), 3:(n2 + 2), 3:(n3 + 2)] .= x
    yp = conv3d_sep(xp, h, h, h; boundary = :zero)
    y = conv3d_sep(x, h, h, h; boundary = :zero)
    @test maximum(abs, yp[3:(n1 + 2), 3:(n2 + 2), 3:(n3 + 2)] .- y) < 1.0e-12
end

@testitem "conv3d_sep! — in-place matches allocating, buffer reuse" tags = [:primitives] begin
    using Random
    Random.seed!(44)
    x = randn(12, 10, 8)
    h = randn(5)
    dst = similar(x); tmp = similar(x)
    conv3d_sep!(dst, x, h, h, h; boundary = :periodic, tmp = tmp)
    @test maximum(abs, dst .- conv3d_sep(x, h, h, h; boundary = :periodic)) < 1.0e-12
    # Reusing the same buffers must give the same answer.
    conv3d_sep!(dst, x, h, h, h; boundary = :periodic, tmp = tmp)
    @test maximum(abs, dst .- conv3d_sep(x, h, h, h; boundary = :periodic)) < 1.0e-12
end

@testitem "conv3d_sep — complex data equals real + i·imag" tags = [:primitives] begin
    using Random
    Random.seed!(45)
    xr = randn(10, 8, 6); xi = randn(10, 8, 6)
    h = randn(5)
    zc = conv3d_sep(complex.(xr, xi), h, h, h; boundary = :periodic)
    zr = conv3d_sep(xr, h, h, h; boundary = :periodic)
    zi = conv3d_sep(xi, h, h, h; boundary = :periodic)
    @test maximum(abs, zc .- complex.(zr, zi)) == 0.0
end

@testitem "conv3d_sep — Float32 stays Float32" tags = [:primitives] begin
    using Random
    Random.seed!(46)
    x = randn(Float32, 8, 8, 8)
    h = Float32[0.25, 0.5, 0.25]
    y = conv3d_sep(x, h, h, h)
    @test eltype(y) == Float32
end

@testitem "conv3d_sep — dimension mismatch throws" tags = [:primitives] begin
    x = randn(8, 8, 8)
    h = [1.0]
    @test_throws DimensionMismatch conv3d_sep!(zeros(8, 8, 7), x, h, h, h)
    @test_throws DimensionMismatch conv3d_sep!(similar(x), x, h, h, h; tmp = zeros(8, 8, 7))
end

@testitem "NSP3D PR — multiple levels" tags = [:pyramid] begin
    using Random
    Random.seed!(50)
    x = randn(16, 16, 16)
    for lv in 1:3
        c, bp = nsp3d_decompose(x, CDF97, lv)
        @test size(c) == size(x)
        @test size(bp) == size(x)
        rec = nsp3d_reconstruct(c, bp, CDF97, lv)
        @test maximum(abs, rec .- x) < 1.0e-12
    end
end

@testitem "NSP3D — prediction has unit DC gain" tags = [:pyramid] begin
    # A constant volume must land entirely in `coarse`; the bandpass must vanish.
    # This is what the per-axis _NSP_SYNTH_SCALE guarantees, and it is the check
    # that catches a wrong 2^N normalisation when going from 2-D to 3-D.
    x = fill(2.5, 16, 16, 16)
    for lv in 1:3
        c, bp = nsp3d_decompose(x, CDF97, lv)
        @test maximum(abs, c .- 2.5) < 1.0e-12
        @test maximum(abs, bp) < 1.0e-12
    end
end

@testitem "NSP3D — exact circular-shift equivariance" tags = [:pyramid] begin
    using Random
    Random.seed!(51)
    x = randn(16, 16, 16)
    s = (3, -5, 2)
    c, bp = nsp3d_decompose(x, CDF97, 2)
    cs, bps = nsp3d_decompose(circshift(x, s), CDF97, 2)
    @test maximum(abs, cs .- circshift(c, s)) < 1.0e-10
    @test maximum(abs, bps .- circshift(bp, s)) < 1.0e-10
end

@testitem "NSP3D in-place round-trip and cached à trous filters" tags = [:pyramid] begin
    using Random
    Random.seed!(52)
    x = randn(16, 16, 16)
    c = similar(x); bp = similar(x); tmp = similar(x); tmp2 = similar(x)
    nsp3d_decompose!(c, bp, x, CDF97, 2; tmp = tmp, tmp2 = tmp2)
    cr, bpr = nsp3d_decompose(x, CDF97, 2)
    @test maximum(abs, c .- cr) < 1.0e-12
    @test maximum(abs, bp .- bpr) < 1.0e-12

    rec = similar(x)
    nsp3d_reconstruct!(rec, c, bp, CDF97, 2; tmp = tmp, tmp2 = tmp2)
    @test maximum(abs, rec .- x) < 1.0e-12

    # Explicitly supplied filters must agree with the internally built ones.
    h_j, g_j = Surfacelets._nsp_atrous_filters(CDF97, 2, Float64)
    c2 = similar(x); bp2 = similar(x)
    nsp3d_decompose!(c2, bp2, x, CDF97, 2; tmp = tmp, tmp2 = tmp2, h_j = h_j, g_j = g_j)
    @test maximum(abs, c2 .- c) < 1.0e-12
    @test maximum(abs, bp2 .- bp) < 1.0e-12
end

@testitem "NSP3D — complex data equals real + i·imag" tags = [:pyramid] begin
    using Random
    Random.seed!(53)
    xr = randn(16, 16, 16); xi = randn(16, 16, 16)
    cc, bc = nsp3d_decompose(complex.(xr, xi), CDF97, 1)
    c1, b1 = nsp3d_decompose(xr, CDF97, 1)
    c2, b2 = nsp3d_decompose(xi, CDF97, 1)
    @test maximum(abs, cc .- complex.(c1, c2)) == 0.0
    @test maximum(abs, bc .- complex.(b1, b2)) == 0.0
end

@testitem "NSP3D — Float32 PR at single precision" tags = [:pyramid] begin
    using Random
    Random.seed!(54)
    x = randn(Float32, 16, 16, 16)
    c, bp = nsp3d_decompose(x, CDF97, 1)
    @test eltype(c) == Float32
    @test eltype(bp) == Float32
    rec = nsp3d_reconstruct(c, bp, CDF97, 1)
    @test maximum(abs, rec .- x) < 1.0f-4
end

@testitem "NSP3D — invalid level throws" tags = [:pyramid] begin
    @test_throws ArgumentError nsp3d_decompose(randn(8, 8, 8), CDF97, 0)
end

@testitem "conv3d_sep_fft — matches direct periodic backend" tags = [:primitives] begin
    using Random
    Random.seed!(60)
    x = randn(32, 24, 20)
    h1 = randn(9); h2 = randn(5); h3 = randn(33)
    direct = conv3d_sep(x, h1, h2, h3; boundary = :periodic, backend = :direct)
    viafft = conv3d_sep(x, h1, h2, h3; boundary = :periodic, backend = :fft)
    @test maximum(abs, direct .- viafft) < 1.0e-11
end

@testitem "conv3d_sep_fft — filter longer than volume aliases like direct" tags = [:primitives] begin
    using Random
    Random.seed!(61)
    x = randn(16, 16, 16)
    h = randn(41)
    direct = conv3d_sep(x, h, h, h; boundary = :periodic, backend = :direct)
    viafft = conv3d_sep(x, h, h, h; boundary = :periodic, backend = :fft)
    @test maximum(abs, direct .- viafft) < 1.0e-10
end

@testitem "conv3d_sep_fft — plan reuse and aliasing dst === src" tags = [:primitives] begin
    using Random
    Random.seed!(62)
    x = randn(Float32, 16, 16, 16)
    h = randn(Float32, 7)
    plan = make_conv3d_fft_plan(size(x), Float32)
    ref = conv3d_sep(x, h, h, h; boundary = :periodic, backend = :direct)
    y = similar(x)
    conv3d_sep_fft!(y, x, h, h, h, plan)
    @test maximum(abs, y .- ref) < 1.0f-4
    # Reusing the plan must not carry state between calls.
    conv3d_sep_fft!(y, x, h, h, h, plan)
    @test maximum(abs, y .- ref) < 1.0f-4
    # In-place aliasing is allowed.
    z = copy(x)
    conv3d_sep_fft!(z, z, h, h, h, plan)
    @test maximum(abs, z .- ref) < 1.0f-4
end

@testitem "conv3d_sep_fft — wrong plan size throws" tags = [:primitives] begin
    x = randn(16, 16, 16)
    h = [1.0]
    plan = make_conv3d_fft_plan((8, 8, 8), Float64)
    @test_throws DimensionMismatch conv3d_sep_fft!(similar(x), x, h, h, h, plan)
    @test_throws ArgumentError make_conv3d_fft_plan((0, 4, 4), Float64)
end

@testitem "conv3d_sep — auto backend agrees with direct for all à trous levels" tags = [:primitives] begin
    using Random
    Random.seed!(63)
    x = randn(32, 32, 32)
    for lv in 1:3
        h = upsample_filter(CDF97.h, 2^(lv - 1))
        auto = conv3d_sep(x, h, h, h; boundary = :periodic)
        direct = conv3d_sep(x, h, h, h; boundary = :periodic, backend = :direct)
        @test maximum(abs, auto .- direct) < 1.0e-11
    end
end

@testitem "NSP3D — FFT and direct backends give the same coefficients" tags = [:pyramid] begin
    using Random
    Random.seed!(64)
    x = randn(32, 32, 32)
    for lv in 1:3
        h_j, g_j = Surfacelets._nsp_atrous_filters(CDF97, lv, Float64)
        plan = make_conv3d_fft_plan(size(x), Float64)
        c = similar(x); bp = similar(x); t1 = similar(x); t2 = similar(x)
        nsp3d_decompose!(c, bp, x, CDF97, lv; tmp = t1, tmp2 = t2, h_j = h_j, g_j = g_j, plan = plan)

        cd = similar(x); bpd = similar(x)
        conv3d_sep!(cd, x, h_j, h_j, h_j; boundary = :periodic, tmp = t2, backend = :direct)
        conv3d_sep!(t1, cd, g_j, g_j, g_j; boundary = :periodic, tmp = t2, backend = :direct)
        @. bpd = x - t1

        @test maximum(abs, c .- cd) < 1.0e-11
        @test maximum(abs, bp .- bpd) < 1.0e-11

        rec = similar(x)
        nsp3d_reconstruct!(rec, c, bp, CDF97, lv; tmp = t1, tmp2 = t2, g_j = g_j, plan = plan)
        @test maximum(abs, rec .- x) < 1.0e-11
    end
end


@testitem "conv3d_sep_fft — complex equals real + i·imag bit-for-bit" tags = [:primitives] begin
    using Random
    Random.seed!(65)
    xr = randn(32, 32, 32); xi = randn(32, 32, 32)
    h = randn(17)
    plan = make_conv3d_fft_plan((32, 32, 32), ComplexF64)
    zc = similar(complex.(xr, xi))
    conv3d_sep_fft!(zc, complex.(xr, xi), h, h, h, plan)
    zr = similar(xr); zi = similar(xi)
    conv3d_sep_fft!(zr, xr, h, h, h, plan)
    conv3d_sep_fft!(zi, xi, h, h, h, plan)
    @test maximum(abs, zc .- complex.(zr, zi)) == 0.0
end

@testitem "conv3d_sep_fft — complex matches direct periodic backend" tags = [:primitives] begin
    using Random
    Random.seed!(66)
    x = complex.(randn(24, 20, 16), randn(24, 20, 16))
    h1 = randn(9); h2 = randn(5); h3 = randn(33)
    direct = conv3d_sep(x, h1, h2, h3; boundary = :periodic, backend = :direct)
    viafft = conv3d_sep(x, h1, h2, h3; boundary = :periodic, backend = :fft)
    @test maximum(abs, direct .- viafft) < 1.0e-11
end

@testitem "conv3d_sep_fft — a real plan serves complex volumes too" tags = [:primitives] begin
    using Random
    Random.seed!(67)
    x = complex.(randn(Float32, 16, 16, 16), randn(Float32, 16, 16, 16))
    h = randn(Float32, 7)
    plan = make_conv3d_fft_plan((16, 16, 16), Float32)   # real plan, no aux yet
    y = similar(x)
    conv3d_sep_fft!(y, x, h, h, h, plan)                 # allocates aux on demand
    ref = conv3d_sep(x, h, h, h; boundary = :periodic, backend = :direct)
    @test maximum(abs, y .- ref) < 1.0f-4
    # Aliased, and reusing the now-warm plan.
    z = copy(x)
    conv3d_sep_fft!(z, z, h, h, h, plan)
    @test maximum(abs, z .- y) == 0.0
end

@testitem "NSP3D — complex volumes take the FFT path" tags = [:pyramid] begin
    using Random
    Random.seed!(68)
    x = complex.(randn(32, 32, 32), randn(32, 32, 32))
    for lv in 1:3
        h_j, g_j = Surfacelets._nsp_atrous_filters(CDF97, lv, Float64)
        @test Surfacelets._nsp3d_plan(x, max(length(h_j), length(g_j))) !== nothing
        c, bp = nsp3d_decompose(x, CDF97, lv)
        @test maximum(abs, nsp3d_reconstruct(c, bp, CDF97, lv) .- x) < 1.0e-11
    end
end
