@testitem "STISurfaceletParams — construction and validation" tags = [:surfacelet] begin
    p = STISurfaceletParams(J = 3, L_array = [1, 2, 2])
    @test eltype(p) == Float64
    @test sti_surfacelet_nsubbands(p, 1) == 6
    @test sti_surfacelet_nsubbands(p, 3) == 12
    @test occursin("STISurfaceletParams", sprint(show, p))
    @test_throws ArgumentError STISurfaceletParams(J = 0, L_array = Int[])
    @test_throws ArgumentError STISurfaceletParams(J = 1, L_array = [-1])
    @test_throws ArgumentError STISurfaceletParams(J = 1, L_array = [1], bo = 3)
end

@testitem "sti_surfacelet — perfect reconstruction" tags = [:surfacelet] begin
    using Random
    rng = Xoshiro(300)
    for (dims, J, L, bo) in (
            ((32, 40, 48), 1, [1], 8),
            ((32, 32, 32), 3, [1, 2, 2], 12),
            ((24, 32, 16), 2, [3, [-1 2 0; 1 -1 1; 0 3 -1]], 4),
            ((16, 16, 16), 2, [0, 0], 12),
        )
        x = randn(rng, dims...)
        p = STISurfaceletParams(J = J, L_array = L, bo = bo)
        pl = make_sti_surfacelet(p, dims)
        c = sti_surfacelet_forward(x, p; plan = pl)
        @test size(c.coarse) == dims
        @test all(j -> length(c.subbands[j]) == sti_surfacelet_nsubbands(p, j), 1:J)
        @test maximum(abs, sti_surfacelet_inverse(c, p; plan = pl) .- x) < 1.0e-12
    end
    # Without a plan.
    x = randn(rng, 16, 16, 16)
    p = STISurfaceletParams(J = 1, L_array = [1])
    @test maximum(abs, sti_surfacelet_inverse(sti_surfacelet_forward(x, p), p) .- x) < 1.0e-12
end

@testitem "sti_surfacelet — redundancy is 1 + 3J, whatever the directions" tags = [:surfacelet] begin
    for (J, L) in ((1, [0]), (1, [4]), (3, [1, 2, 3]), (3, [3, 3, 3]))
        pl = make_sti_surfacelet(STISurfaceletParams(J = J, L_array = L, bo = 4), (64, 64, 64))
        @test sti_surfacelet_redundancy(pl) == 1 + 3J
    end
end

@testitem "sti_surfacelet — the pyramid is a tight frame" tags = [:surfacelet] begin
    pl = make_sti_surfacelet(STISurfaceletParams(J = 3, L_array = [1, 1, 1]), (20, 24, 28))
    L = [Surfacelets._sti_lowpass(pl.low[j]) for j in 1:3]
    prior = ones(size(L[1]))
    total = zeros(size(L[1]))
    for j in 1:3
        total .+= prior .^ 2 .* (1 .- L[j] .^ 2)
        prior .*= L[j]
    end
    total .+= prior .^ 2
    @test maximum(abs, total .- 1) < 1.0e-14
end

@testitem "sti_surfacelet — shift invariance" tags = [:surfacelet] begin
    using Random
    x = randn(Xoshiro(301), 32, 40, 48)
    p = STISurfaceletParams(J = 2, L_array = [2, 3])
    pl = make_sti_surfacelet(p, size(x))
    a = sti_surfacelet_forward(x, p; plan = pl)
    # The coarse band is exactly invariant under any circular shift.
    s = (3, -5, 7)
    b = sti_surfacelet_forward(circshift(x, s), p; plan = pl)
    @test maximum(abs, circshift(a.coarse, s) .- b.coarse) < 1.0e-13
    # A directional subband of hourglass K is decimated by 2^l along each other
    # axis, so a shift on that lattice moves its coefficients exactly.
    s = (8, 8, 16)
    b = sti_surfacelet_forward(circshift(x, s), p; plan = pl)
    for j in 1:2
        i = 0
        for K in 1:3
            lv = p.levels[j][K]
            for _ in 1:(1 << sum(lv))
                i += 1
                sh = ntuple(d -> d == K ? s[d] : s[d] ÷ (1 << lv[d]), 3)
                @test maximum(abs, circshift(a.subbands[j][i], sh) .- b.subbands[j][i]) <
                    1.0e-12 * maximum(abs, a.subbands[j][i])
            end
        end
    end
end

@testitem "sti_surfacelet — a plane wave lands in its own hourglass at its scale" tags = [:surfacelet] begin
    n = 32
    p = STISurfaceletParams(J = 2, L_array = [2, 2])
    pl = make_sti_surfacelet(p, (n, n, n))
    # |k|∞ = 11 sits in scale 1 (π/2 … π), 5 in scale 2 (π/4 … π/2).
    for (k, j, K) in (((11, 3, 2), 1, 1), ((2, -11, 4), 1, 2), ((1, 2, 5), 2, 3))
        v = [
            cospi(2 * (k[1] * (I[1] - 1) + k[2] * (I[2] - 1) + k[3] * (I[3] - 1)) / n)
                for I in CartesianIndices((n, n, n))
        ]
        c = sti_surfacelet_forward(v, p; plan = pl)
        e = [[sum(abs2, b) for b in s] for s in c.subbands]
        @test argmax(sum.(e)) == j
        @test (argmax(e[j]) - 1) ÷ 4 + 1 == K
    end
end

@testitem "sti_surfacelet — Float32 and validation" tags = [:surfacelet] begin
    using Random
    x = randn(Xoshiro(302), Float32, 16, 16, 16)
    p = STISurfaceletParams(J = 1, L_array = [1], precision = Float32)
    pl = make_sti_surfacelet(p, size(x), Float32)
    c = sti_surfacelet_forward(x, p; plan = pl)
    @test eltype(c) == Float32
    @test maximum(abs, sti_surfacelet_inverse(c, p; plan = pl) .- x) < 1.0f-4
    @test make_sti_surfacelet(p, 16, 16, 16).dims == (16, 16, 16)

    q = STISurfaceletParams(J = 1, L_array = [1])
    plq = make_sti_surfacelet(q, (16, 16, 16))
    @test_throws ArgumentError make_sti_surfacelet(q, (15, 16, 16))
    @test_throws ArgumentError make_sti_surfacelet(STISurfaceletParams(J = 1, L_array = [9]), (16, 16, 16))
    @test_throws DimensionMismatch sti_surfacelet_forward(randn(8, 8, 8), q; plan = plq)
    cq = sti_surfacelet_forward(randn(16, 16, 16), q; plan = plq)
    @test_throws ArgumentError sti_surfacelet_inverse(cq, STISurfaceletParams(J = 2, L_array = [1, 1]))
    @test_throws DimensionMismatch sti_surfacelet_inverse(
        SurfaceletCoefficients(cq.dims, zeros(4, 4, 4), cq.subbands), q; plan = plq
    )
    @test_throws DimensionMismatch sti_surfacelet_inverse(
        SurfaceletCoefficients(cq.dims, cq.coarse, [cq.subbands[1][1:2]]), q; plan = plq
    )
end
