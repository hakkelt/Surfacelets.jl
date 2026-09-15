@testitem "ndfb — subband counting and index round trip" tags = [:directional] begin
    for L in 0:3
        fb = make_ndfb((8, 8, 8), Float64; levels = L, order = 4)
        @test ndfb_nsubbands(fb) == 3 * 2^L
        na, nb = 1 << fb.levels[1], 1 << fb.levels[2]
        @test na * nb == 2^L
        seen = Int[]
        for i in 1:3, p in 1:na, q in 1:nb
            idx = ndfb_subband_index(fb, i, p, q)
            push!(seen, idx)
            @test ndfb_subband_triple(fb, idx) == (i, p, q)
        end
        @test sort(seen) == collect(1:ndfb_nsubbands(fb))
    end
    fb = make_ndfb((8, 8, 8), Float64; levels = 2, order = 4)
    @test_throws ArgumentError ndfb_subband_index(fb, 4, 1, 1)
    @test_throws ArgumentError ndfb_subband_index(fb, 1, 99, 1)
    @test_throws ArgumentError ndfb_subband_triple(fb, 0)
    @test_throws ArgumentError make_ndfb((8, 8, 8), Float64; levels = -1)
end

@testitem "ndfb — weights are a partition of unity" tags = [:directional] begin
    for L in 0:3
        fb = make_ndfb((10, 12, 8), Float64; levels = L, order = 6)
        total = sum(ndfb_weight(fb, i) for i in 1:ndfb_nsubbands(fb))
        @test maximum(abs, total .- 1) < 1.0e-13
        # Tight by default, so it is the squares of the masks that sum to 1.
        squares = sum(ndfb_mask(fb, i) .^ 2 for i in 1:ndfb_nsubbands(fb))
        @test maximum(abs, squares .- 1) < 1.0e-13
    end
end

@testitem "ndfb — tight masks are a Parseval frame" tags = [:directional] begin
    using Random
    x = randn(Xoshiro(81), 16, 16, 16)
    for L in 0:3
        fb = make_ndfb(size(x), Float64; levels = L, order = 6)
        s = ndfb_decompose(x, fb)
        @test abs(sum(sum(abs2, b) for b in s) / sum(abs2, x) - 1) < 1.0e-12
        @test maximum(abs, ndfb_reconstruct(s, fb) .- x) < 1.0e-12
    end
    loose = make_ndfb(size(x), Float64; levels = 2, order = 6, tight = false)
    sl = ndfb_decompose(x, loose)
    @test sum(sum(abs2, b) for b in sl) / sum(abs2, x) < 0.9
    @test maximum(abs, ndfb_reconstruct(sl, loose) .- x) < 1.0e-12
end

@testitem "ndfb — masks are even in frequency" tags = [:directional] begin
    # A wedge must give the same value to ω and -ω, or the masked spectrum is
    # not Hermitian and the real subband loses information. On the DFT grid the
    # trap is that an even-length axis fixes its Nyquist bin under ω -> -ω.
    # 12 has `2π(n/2)/n == π` exactly; 26 rounds a ulp *past* π (so folding sends
    # the Nyquist bin to ≈ -π) and 30 a ulp short of it. Identifying that bin by
    # value rather than by index misses both, and the tight bank then silently
    # loses perfect reconstruction — so all three sizes are pinned here.
    for n in (12, 26, 30)
        fb = make_ndfb((n, n, n), Float64; levels = 2, order = 6)
        neg(i) = mod(1 - i, n) + 1
        for idx in 1:ndfb_nsubbands(fb)
            m = ndfb_mask(fb, idx)
            # On the two self-conjugate planes of the rfft grid the mask has to
            # be even in the remaining two axes on its own.
            for i in (1, n ÷ 2 + 1), j in 1:n, k in 1:n
                @test m[i, j, k] ≈ m[i, neg(j), neg(k)] atol = 1.0e-13
            end
        end
    end
end

@testitem "ndfb — tight PR survives a Nyquist bin that is not exactly π" tags = [:directional] begin
    using Random
    # `dft_frequencies` computes `2π(k-1)/n`, which lands an ulp off `π` at the
    # Nyquist bin for about half of all even `n`. The wedge masks must still be
    # even there, or the `√W` synthesis of `tight = true` is lossy.
    for dims in ((30, 16, 16), (16, 30, 16), (16, 16, 30), (26, 26, 26), (60, 16, 16))
        x = randn(Xoshiro(83), dims...)
        for L in 1:3
            fb = make_ndfb(dims, Float64; levels = L, order = 6)
            s = ndfb_decompose(x, fb)
            @test maximum(abs, ndfb_reconstruct(s, fb) .- x) < 1.0e-11
            @test abs(sum(sum(abs2, b) for b in s) / sum(abs2, x) - 1) < 1.0e-11
        end
    end
end

@testitem "ndfb — wedges of one hourglass sum to that hourglass weight" tags = [:directional] begin
    fb = make_ndfb((12, 12, 12), Float64; levels = 3, order = 6)
    na, nb = 1 << fb.levels[1], 1 << fb.levels[2]
    for i in 1:3
        acc = sum(
            ndfb_weight(fb, ndfb_subband_index(fb, i, p, q))
                for p in 1:na, q in 1:nb
        )
        @test maximum(abs, acc .- hourglass_weight(fb.fb, i)) < 1.0e-13
    end
end

@testitem "ndfb — perfect reconstruction" tags = [:directional] begin
    using Random
    Random.seed!(80)
    for dims in ((16, 16, 16), (12, 20, 8))
        x = randn(dims...)
        for L in 0:3, tr in (0.0, 0.5, 1.0)
            fb = make_ndfb(dims, Float64; levels = L, order = 6, transition = tr)
            s = ndfb_decompose(x, fb)
            @test length(s) == ndfb_nsubbands(fb)
            @test maximum(abs, ndfb_reconstruct(s, fb) .- x) < 1.0e-12
        end
    end
end

@testitem "ndfb — plane waves land in the predicted wedge" tags = [:directional] begin
    n = 32
    fb = make_ndfb((n, n, n), Float64; levels = 4, order = 12, transition = 0.5)
    la, lb = fb.levels
    transverse(i) = i == 1 ? (2, 3) : (i == 2 ? (1, 3) : (1, 2))
    function predict(k)
        i = argmax(abs.(k))
        a, b = transverse(i)
        p = clamp(floor(Int, (k[a] / k[i] + 1) / (2 / (1 << la))) + 1, 1, 1 << la)
        q = clamp(floor(Int, (k[b] / k[i] + 1) / (2 / (1 << lb))) + 1, 1, 1 << lb)
        return ndfb_subband_index(fb, i, p, q)
    end
    # Frequencies chosen away from wedge and hourglass boundaries: no two
    # magnitudes equal, and neither ratio lands on a wedge edge.
    for k in ([2, 5, 9], [5, 9, -12], [-2, 3, -13], [-13, -9, 7], [9, 7, -12], [3, 11, -7])
        v = [
            cospi(2 * (k[1] * (I[1] - 1) + k[2] * (I[2] - 1) + k[3] * (I[3] - 1)) / n)
                for I in CartesianIndices((n, n, n))
        ]
        e = zeros(ndfb_nsubbands(fb))
        ndfb_decompose(v, fb) do idx, band
            e[idx] = sum(abs2, band)
            return nothing
        end
        @test argmax(e) == predict(k)
        # The winning wedge must dominate, not merely edge out its neighbours.
        @test e[argmax(e)] / sum(e) > 0.3
    end
end

@testitem "ndfb — streaming callback matches the materialised subbands" tags = [:directional] begin
    using Random
    Random.seed!(81)
    x = randn(16, 16, 16)
    fb = make_ndfb(size(x), Float64; levels = 2, order = 6)
    ref = ndfb_decompose(x, fb)
    seen = Int[]
    ndfb_decompose(x, fb) do idx, band
        push!(seen, idx)
        @test maximum(abs, band .- ref[idx]) == 0.0
        return nothing
    end
    @test seen == collect(1:ndfb_nsubbands(fb))
end

@testitem "ndfb — in-place decomposition" tags = [:directional] begin
    using Random
    Random.seed!(82)
    x = randn(16, 16, 16)
    fb = make_ndfb(size(x), Float64; levels = 2, order = 6)
    dst = [similar(x) for _ in 1:ndfb_nsubbands(fb)]
    ndfb_decompose!(dst, x, fb)
    ref = ndfb_decompose(x, fb)
    @test all(i -> maximum(abs, dst[i] .- ref[i]) == 0.0, eachindex(ref))
    @test_throws DimensionMismatch ndfb_decompose!(dst[1:2], x, fb)
    @test_throws DimensionMismatch ndfb_reconstruct!(similar(x), dst[1:2], fb)
end

@testitem "ndfb — exact circular-shift equivariance" tags = [:directional] begin
    using Random
    Random.seed!(83)
    x = randn(16, 16, 16)
    fb = make_ndfb(size(x), Float64; levels = 2, order = 6)
    s = (5, -3, 2)
    a = ndfb_decompose(x, fb)
    b = ndfb_decompose(circshift(x, s), fb)
    for i in eachindex(a)
        @test maximum(abs, b[i] .- circshift(a[i], s)) < 1.0e-10
    end
end

@testitem "ndfb — complex equals real + i·imag bit-for-bit" tags = [:directional] begin
    using Random
    Random.seed!(84)
    xr = randn(16, 16, 16); xi = randn(16, 16, 16)
    fb = make_ndfb(size(xr), ComplexF64; levels = 2, order = 6)
    sc = ndfb_decompose(complex.(xr, xi), fb)
    sr = ndfb_decompose(xr, fb)
    si = ndfb_decompose(xi, fb)
    for i in eachindex(sc)
        @test maximum(abs, sc[i] .- complex.(sr[i], si[i])) == 0.0
    end
    @test maximum(abs, ndfb_reconstruct(sc, fb) .- complex.(xr, xi)) < 1.0e-12
end

@testitem "ndfb — Float32" tags = [:directional] begin
    using Random
    Random.seed!(85)
    x = randn(Float32, 16, 16, 16)
    fb = make_ndfb(size(x), Float32; levels = 2, order = 6)
    s = ndfb_decompose(x, fb)
    @test all(b -> eltype(b) == Float32, s)
    @test maximum(abs, ndfb_reconstruct(s, fb) .- x) < 1.0f-4
end

@testitem "ndfb — dimension validation" tags = [:directional] begin
    fb = make_ndfb((8, 8, 8), Float64; levels = 1, order = 4)
    @test_throws DimensionMismatch ndfb_decompose(randn(8, 8, 4), fb)
    @test_throws DimensionMismatch ndfb_decompose((i, b) -> nothing, randn(4, 8, 8), fb)
end
