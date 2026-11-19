@testitem "NSSurfaceletParams — construction and validation" tags = [:ns_surfacelet] begin
    p = NSSurfaceletParams(J = 3, L_array = [1, 2, 2])
    @test p.J == 3
    @test p.L_array == [1, 2, 2]
    @test eltype(p) == Float64
    @test ns_surfacelet_nsubbands(p, 1) == 6
    @test ns_surfacelet_nsubbands(p, 2) == 12
    @test occursin("NSSurfaceletParams", sprint(show, p))
    @test_throws ArgumentError NSSurfaceletParams(J = 0, L_array = Int[])
    @test_throws ArgumentError NSSurfaceletParams(J = 2, L_array = [1])
    @test_throws ArgumentError NSSurfaceletParams(J = 1, L_array = [-1])
    @test_throws ArgumentError NSSurfaceletParams(J = 1, L_array = [1], order = 0)
    @test_throws ArgumentError NSSurfaceletParams(J = 1, L_array = [1], poly = -1)
    @test_throws ArgumentError NSSurfaceletParams(J = 1, L_array = [1], transition = -1)
end

@testitem "ns_surfacelet — perfect reconstruction" tags = [:ns_surfacelet] begin
    using Random
    Random.seed!(90)
    for dims in ((16, 16, 16), (12, 20, 8))
        x = randn(dims...)
        for (J, L) in ((1, [0]), (2, [1, 2]), (3, [1, 1, 2]))
            p = NSSurfaceletParams(J = J, L_array = L, order = 6)
            c = ns_surfacelet_forward(x, p)
            @test length(c.subbands) == J
            @test all(j -> length(c.subbands[j]) == 3 * 2^L[j], 1:J)
            @test all(j -> all(b -> size(b) == dims, c.subbands[j]), 1:J)
            @test size(c.coarse) == dims
            @test maximum(abs, ns_surfacelet_inverse(c, p) .- x) < 1.0e-12
        end
    end
end

@testitem "ns_surfacelet — shift invariance" tags = [:ns_surfacelet] begin
    using Random
    Random.seed!(91)
    x = randn(16, 16, 16)
    p = NSSurfaceletParams(J = 2, L_array = [1, 2], order = 6)
    s = (3, -2, 5)
    a = ns_surfacelet_forward(x, p)
    b = ns_surfacelet_forward(circshift(x, s), p)
    @test maximum(abs, b.coarse .- circshift(a.coarse, s)) < 1.0e-10
    for j in 1:2, k in eachindex(a.subbands[j])
        @test maximum(abs, b.subbands[j][k] .- circshift(a.subbands[j][k], s)) < 1.0e-10
    end
end

@testitem "ns_surfacelet — a constant volume stays in the coarse residual" tags = [:ns_surfacelet] begin
    x = fill(1.75, 16, 16, 16)
    p = NSSurfaceletParams(J = 2, L_array = [1, 1], order = 6)
    c = ns_surfacelet_forward(x, p)
    @test maximum(abs, c.coarse .- 1.75) < 1.0e-12
    for j in 1:2, b in c.subbands[j]
        @test maximum(abs, b) < 1.0e-12
    end
end

@testitem "ns_surfacelet — streaming forward matches the materialised transform" tags = [:ns_surfacelet] begin
    using Random
    Random.seed!(92)
    x = randn(16, 16, 16)
    p = NSSurfaceletParams(J = 2, L_array = [1, 2], order = 6)
    ref = ns_surfacelet_forward(x, p)
    seen = Tuple{Int, Int}[]
    coarse = ns_surfacelet_forward(x, p) do j, k, band
        push!(seen, (j, k))
        @test maximum(abs, band .- ref.subbands[j][k]) == 0.0
    end
    @test maximum(abs, coarse .- ref.coarse) == 0.0
    @test seen == [(j, k) for j in 1:2 for k in 1:(3 * 2^p.L_array[j])]
end

@testitem "ns_surfacelet — streaming inverse matches" tags = [:ns_surfacelet] begin
    using Random
    Random.seed!(93)
    x = randn(16, 16, 16)
    p = NSSurfaceletParams(J = 2, L_array = [1, 2], order = 6)
    c = ns_surfacelet_forward(x, p)
    r = ns_surfacelet_inverse(c.coarse, p) do j, k
        return c.subbands[j][k]
    end
    @test maximum(abs, r .- x) < 1.0e-12
    @test_throws DimensionMismatch ns_surfacelet_inverse(c.coarse, p) do j, k
        return zeros(4, 4, 4)
    end
end

@testitem "ns_surfacelet — complex equals real + i·imag bit-for-bit" tags = [:ns_surfacelet] begin
    using Random
    Random.seed!(94)
    xr = randn(16, 16, 16); xi = randn(16, 16, 16)
    p = NSSurfaceletParams(J = 2, L_array = [1, 1], order = 6)
    cc = ns_surfacelet_forward(complex.(xr, xi), p)
    cr = ns_surfacelet_forward(xr, p)
    ci = ns_surfacelet_forward(xi, p)
    @test maximum(abs, cc.coarse .- complex.(cr.coarse, ci.coarse)) == 0.0
    for j in 1:2, k in eachindex(cc.subbands[j])
        @test maximum(abs, cc.subbands[j][k] .- complex.(cr.subbands[j][k], ci.subbands[j][k])) == 0.0
    end
    @test maximum(abs, ns_surfacelet_inverse(cc, p) .- complex.(xr, xi)) < 1.0e-12
end

@testitem "ns_surfacelet — Float32" tags = [:ns_surfacelet] begin
    using Random
    Random.seed!(95)
    x = randn(Float32, 16, 16, 16)
    p = NSSurfaceletParams(J = 2, L_array = [1, 1], order = 6)
    c = ns_surfacelet_forward(x, p)
    @test eltype(c) == Float32
    @test eltype(c.coarse) == Float32
    @test maximum(abs, ns_surfacelet_inverse(c, p) .- x) < 1.0f-4
end

@testitem "ns_surfacelet — workspace reuse is bit-identical and validated" tags = [:ns_surfacelet] begin
    using Random
    Random.seed!(96)
    x = randn(16, 16, 16)
    p = NSSurfaceletParams(J = 2, L_array = [1, 2], order = 6)
    ws = make_ns_surfacelet_workspace(p, (16, 16, 16), Float64)
    a = ns_surfacelet_forward(x, p)
    b = ns_surfacelet_forward(x, p; workspace = ws)
    c = ns_surfacelet_forward(x, p; workspace = ws)   # second call must not drift
    @test maximum(abs, b.coarse .- a.coarse) == 0.0
    @test maximum(abs, c.coarse .- a.coarse) == 0.0
    for j in 1:2, k in eachindex(a.subbands[j])
        @test maximum(abs, b.subbands[j][k] .- a.subbands[j][k]) == 0.0
        @test maximum(abs, c.subbands[j][k] .- a.subbands[j][k]) == 0.0
    end
    @test maximum(abs, ns_surfacelet_inverse(b, p; workspace = ws) .- x) < 1.0e-12

    @test_throws DimensionMismatch ns_surfacelet_forward(randn(8, 8, 8), p; workspace = ws)
    @test_throws ArgumentError ns_surfacelet_forward(complex.(x, x), p; workspace = ws)
    wsj = make_ns_surfacelet_workspace(NSSurfaceletParams(J = 1, L_array = [1]), (16, 16, 16), Float64)
    @test_throws ArgumentError ns_surfacelet_forward(x, p; workspace = wsj)
    # Per-scale directional depth and the `tight` normalisation are baked into
    # the banks, so a workspace that disagrees with `params` must be rejected
    # rather than silently overriding it.
    wsl = make_ns_surfacelet_workspace(NSSurfaceletParams(J = 2, L_array = [2, 2], order = 6), (16, 16, 16), Float64)
    @test_throws ArgumentError ns_surfacelet_forward(x, p; workspace = wsl)
    wst = make_ns_surfacelet_workspace(
        NSSurfaceletParams(J = 2, L_array = [1, 2], order = 6, tight = false),
        (16, 16, 16), Float64
    )
    @test_throws ArgumentError ns_surfacelet_forward(x, p; workspace = wst)
end

@testitem "ns_surfacelet — show and coefficient container" tags = [:ns_surfacelet] begin
    using Random
    Random.seed!(97)
    x = randn(8, 8, 8)
    p = NSSurfaceletParams(J = 1, L_array = [1], order = 4)
    c = ns_surfacelet_forward(x, p)
    s = sprint(show, c)
    @test occursin("NSSurfaceletCoefficients", s)
    @test occursin("J=1", s)
end

@testitem "ns_surfacelet — a Parseval frame: the inverse is the adjoint" tags = [:ns_surfacelet] begin
    using Random, LinearAlgebra
    Random.seed!(98)
    # ‖Sx‖ = ‖x‖ and ⟨Sx, c⟩ = ⟨x, S⁻¹c⟩ for random coefficients c: the inverse is
    # the adjoint.  Odd and short axes included (a cardiac axis has 8 bins).
    inner(a, b) = dot(a.coarse, b.coarse) +
        sum(dot(u, v) for (su, sv) in zip(a.subbands, b.subbands) for (u, v) in zip(su, sv))
    for dims in ((16, 16, 16), (15, 20, 8)), (J, L) in ((1, [1]), (3, [1, 1, 2]))
        p = NSSurfaceletParams(J = J, L_array = L, order = 6)
        x = randn(dims...)
        c = ns_surfacelet_forward(x, p)
        @test inner(c, c) ≈ sum(abs2, x) rtol = 1.0e-12
        r = deepcopy(c)
        r.coarse .= randn(dims...)
        foreach(sj -> foreach(b -> b .= randn(dims...), sj), r.subbands)
        @test inner(c, r) ≈ dot(x, ns_surfacelet_inverse(r, p)) rtol = 1.0e-12
        z = complex.(x, randn(dims...))
        cz = ns_surfacelet_forward(z, p)
        @test real(inner(cz, cz)) ≈ sum(abs2, z) rtol = 1.0e-12
    end
end

@testitem "ns_surfacelet — a scale with an empty bandpass is refused" tags = [:ns_surfacelet] begin
    using Random
    # At 8 bins the third level's lowpass passes DC alone, which the fourth keeps
    # whole: scale 4 would be identically zero.
    p = NSSurfaceletParams(J = 4, L_array = [0, 0, 0, 0], order = 4)
    @test_throws ArgumentError make_ns_surfacelet_workspace(p, (8, 8, 8))
    # A short axis alone is not refused while the others still split the band.
    x = randn(Xoshiro(3), 32, 32, 8)
    q = NSSurfaceletParams(J = 3, L_array = [0, 1, 1], order = 4)
    c = ns_surfacelet_forward(x, q)
    @test all(j -> maximum(abs, c.subbands[j][1]) > 0, 1:3)
    @test maximum(abs, ns_surfacelet_inverse(c, q) .- x) < 1.0e-12
end

@testitem "ns_surfacelet — uncached bandpass matches the cache" tags = [:ns_surfacelet] begin
    using Random
    for (Td, tight) in ((Float64, true), (Float64, false), (ComplexF64, true))
        x = randn(Xoshiro(7), Td, 16, 12, 20)
        p = NSSurfaceletParams(J = 2, L_array = [1, 2], order = 6, tight = tight)
        ws = make_ns_surfacelet_workspace(p, size(x), Td)
        wu = make_ns_surfacelet_workspace(p, size(x), Td; cache_bandpass = false)
        @test length(ws.band) == 2 && isempty(wu.band)
        c = ns_surfacelet_forward(x, p; workspace = ws)
        d = ns_surfacelet_forward(x, p; workspace = wu)
        @test d.coarse ≈ c.coarse
        @test all(d.subbands[j][k] ≈ c.subbands[j][k] for j in 1:2 for k in eachindex(c.subbands[j]))
        @test maximum(abs, ns_surfacelet_inverse(d, p; workspace = wu) .- x) < 1.0e-12
        for threaded in (true, false)
            @test ns_surfacelet_inverse(d, p; workspace = wu, threaded) ≈ ns_surfacelet_inverse(c, p; workspace = ws, threaded)
        end
    end
end

@testitem "ns_surfacelet — adapting a workspace to the host keeps it" tags = [:ns_surfacelet] begin
    using Adapt, Random
    x = randn(Xoshiro(11), 16, 16, 16)
    p = NSSurfaceletParams(J = 2, L_array = [1, 1], order = 6)
    for cache in (true, false)
        ws = make_ns_surfacelet_workspace(p, size(x); cache_bandpass = cache)
        wh = Adapt.adapt(Array, ws)
        @test length(wh.engine.buffers) == length(ws.engine.buffers)
        @test length(wh.band) == length(ws.band)
        @test ns_surfacelet_forward(x, p; workspace = wh).subbands ≈ ns_surfacelet_forward(x, p; workspace = ws).subbands
    end
end

@testitem "ns_surfacelet — arrays the FFT plans cannot use directly" tags = [:ns_surfacelet] begin
    using Random
    # Real subbands and outputs that are not plain arrays of the plans' type go
    # through the workspace's one-task path; an integer volume is copied into a
    # real buffer first.  Both give the threaded path's result exactly.
    x = randn(Xoshiro(13), 12, 16, 10)
    p = NSSurfaceletParams(J = 2, L_array = [1, 2], order = 6)
    ws = make_ns_surfacelet_workspace(p, size(x))
    c = ns_surfacelet_forward(x, p; workspace = ws)
    viewed(a) = view(similar(a), :, :, :)
    cv = NSSurfaceletCoefficients(viewed(c.coarse), [map(viewed, s) for s in c.subbands])
    ns_surfacelet_forward!(cv, x, p; workspace = ws)
    @test cv.coarse == c.coarse
    @test all(cv.subbands[j][k] == c.subbands[j][k] for j in 1:2 for k in eachindex(c.subbands[j]))
    y = ns_surfacelet_inverse(c, p; workspace = ws)
    @test ns_surfacelet_inverse(cv, p; workspace = ws) ≈ y
    yv = view(similar(x), :, :, :)
    @test ns_surfacelet_inverse!(yv, c, p; workspace = ws) == y
    @test maximum(abs, y .- x) < 1.0e-12
    xi = rand(Xoshiro(14), -3:3, size(x)...)
    ci = ns_surfacelet_forward(xi, p; workspace = ws)
    cf = ns_surfacelet_forward(float.(xi), p; workspace = ws)
    @test ci.coarse == cf.coarse
    @test all(ci.subbands[j][k] == cf.subbands[j][k] for j in 1:2 for k in eachindex(cf.subbands[j]))
end
