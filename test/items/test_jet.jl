@testitem "@test_call Surfacelets public API" tags = [:jet] begin
    using JET, Random
    Random.seed!(100)

    dims = (16, 16, 16)
    x = randn(dims...)
    dst = similar(x)

    # ── Types / constructors ─────────────────────────────────────────────────
    @test_call target_modules = (Surfacelets,) NSurfaceletParams(J = 2, L_array = [1, 1], order = 6)
    @test_call target_modules = (Surfacelets,) SurfaceletParams(J = 2, L_array = [1, 1])

    # ── Primitives ───────────────────────────────────────────────────────────
    h = [0.25, 0.5, 0.25]
    @test_call target_modules = (Surfacelets,) conv3d_sep(x, h, h, h)
    @test_call target_modules = (Surfacelets,) conv3d_sep!(dst, x, h, h, h)
    @test_call target_modules = (Surfacelets,) make_conv3d_fft_plan(dims, Float64)
    plan = make_conv3d_fft_plan(dims, Float64)
    @test_call target_modules = (Surfacelets,) conv3d_sep_fft!(dst, x, h, h, h, plan)

    # ── Pyramid ──────────────────────────────────────────────────────────────
    @test_call target_modules = (Surfacelets,) nsp3d_decompose(x, CDF97, 1)
    c, bp = nsp3d_decompose(x, CDF97, 1)
    @test_call target_modules = (Surfacelets,) nsp3d_reconstruct(c, bp, CDF97, 1)

    # ── Directional FB ───────────────────────────────────────────────────────
    @test_call target_modules = (Surfacelets,) make_hourglass(dims, Float64; order = 6)
    hg = make_hourglass(dims, Float64; order = 6)
    @test_call target_modules = (Surfacelets,) hourglass_decompose(x, hg)
    hs = hourglass_decompose(x, hg)
    @test_call target_modules = (Surfacelets,) hourglass_reconstruct(hs, hg)

    @test_call target_modules = (Surfacelets,) make_ndfb(dims, Float64; levels = 1, order = 6)
    fb = make_ndfb(dims, Float64; levels = 1, order = 6)
    @test_call target_modules = (Surfacelets,) ndfb_nsubbands(fb)
    @test_call target_modules = (Surfacelets,) ndfb_decompose(x, fb)
    sbs = ndfb_decompose(x, fb)
    @test_call target_modules = (Surfacelets,) ndfb_reconstruct(sbs, fb)

    # ── Transforms ───────────────────────────────────────────────────────────
    p = NSurfaceletParams(J = 1, L_array = [1], order = 6)
    @test_call target_modules = (Surfacelets,) make_nsurfacelet_workspace(p, dims, Float64)
    ws = make_nsurfacelet_workspace(p, dims, Float64)
    @test_call target_modules = (Surfacelets,) nsurfacelet_forward(x, p; workspace = ws)
    coeffs = nsurfacelet_forward(x, p; workspace = ws)
    @test_call target_modules = (Surfacelets,) nsurfacelet_inverse(coeffs, p; workspace = ws)
    @test_call target_modules = (Surfacelets,) nsurfacelet_inverse!(dst, coeffs, p; workspace = ws)

    sp = SurfaceletParams(J = 1, L_array = [1])
    @test_call target_modules = (Surfacelets,) make_surfacelet(sp, dims, Float64)
    spl = make_surfacelet(sp, dims, Float64)
    @test_call target_modules = (Surfacelets,) surfacelet_forward(x, sp; plan = spl)
    sc = surfacelet_forward(x, sp; plan = spl)
    @test_call target_modules = (Surfacelets,) surfacelet_inverse(sc, sp; plan = spl)
    @test_call target_modules = (Surfacelets,) surfacelet_redundancy(spl)
end

@testitem "@test_opt Surfacelets core functions" tags = [:jet] begin
    using JET, Random
    Random.seed!(101)

    dims = (16, 16, 16)
    x = randn(dims...)
    dst = similar(x)
    h = [0.25, 0.5, 0.25]

    # ── Primitives ───────────────────────────────────────────────────────────
    @test_opt target_modules = (Surfacelets,) conv3d_sep(x, h, h, h)
    @test_opt target_modules = (Surfacelets,) conv3d_sep!(dst, x, h, h, h)

    # ── Pyramid ──────────────────────────────────────────────────────────────
    @test_opt target_modules = (Surfacelets,) nsp3d_decompose(x, CDF97, 1)
    c, bp = nsp3d_decompose(x, CDF97, 1)
    @test_opt target_modules = (Surfacelets,) nsp3d_reconstruct(c, bp, CDF97, 1)

    # ── Directional FB ───────────────────────────────────────────────────────
    hg = make_hourglass(dims, Float64; order = 6)
    @test_opt target_modules = (Surfacelets,) hourglass_decompose(x, hg)
    hs = hourglass_decompose(x, hg)
    @test_opt target_modules = (Surfacelets,) hourglass_reconstruct(hs, hg)

    fb = make_ndfb(dims, Float64; levels = 1, order = 6)
    @test_opt target_modules = (Surfacelets,) ndfb_decompose(x, fb)

    # ── Transforms ───────────────────────────────────────────────────────────
    p = NSurfaceletParams(J = 1, L_array = [1], order = 6)
    ws = make_nsurfacelet_workspace(p, dims, Float64)
    @test_opt target_modules = (Surfacelets,) nsurfacelet_forward(x, p; workspace = ws)
    coeffs = nsurfacelet_forward(x, p; workspace = ws)
    @test_opt target_modules = (Surfacelets,) nsurfacelet_inverse!(dst, coeffs, p; workspace = ws)
end
