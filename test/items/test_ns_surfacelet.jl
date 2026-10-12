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
