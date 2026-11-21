@testitem "@test_call Surfacelets public API" tags = [:jet] begin
    using Surfacelets: make_hourglass, hourglass_decompose, hourglass_reconstruct,
        make_ndfb, ndfb_nsubbands, ndfb_decompose, ndfb_reconstruct
    using JET, Random
    Random.seed!(100)

    dims = (16, 16, 16)
    x = randn(dims...)
    dst = similar(x)

    # ── Types / constructors ─────────────────────────────────────────────────
    @test_call target_modules = (Surfacelets,) NSSurfacelet(scales = 2, direction_levels = [1, 1], bernstein_order = 6)
    @test_call target_modules = (Surfacelets,) Surfacelet(scales = 2, direction_levels = [1, 1])

    # ── Directional FB ───────────────────────────────────────────────────────
    @test_call target_modules = (Surfacelets,) make_hourglass(Float64, dims; bernstein_order = 6)
    hg = make_hourglass(Float64, dims; bernstein_order = 6)
    @test_call target_modules = (Surfacelets,) hourglass_decompose(x, hg)
    hs = hourglass_decompose(x, hg)
    @test_call target_modules = (Surfacelets,) hourglass_reconstruct(hs, hg)

    @test_call target_modules = (Surfacelets,) make_ndfb(Float64, dims; levels = 1, bernstein_order = 6)
    fb = make_ndfb(Float64, dims; levels = 1, bernstein_order = 6)
    @test_call target_modules = (Surfacelets,) ndfb_nsubbands(fb)
    @test_call target_modules = (Surfacelets,) ndfb_decompose(x, fb)
    sbs = ndfb_decompose(x, fb)
    @test_call target_modules = (Surfacelets,) ndfb_reconstruct(sbs, fb)

    # ── Transforms ───────────────────────────────────────────────────────────
    t = NSSurfacelet(scales = 1, direction_levels = [1], bernstein_order = 6)
    @test_call target_modules = (Surfacelets,) make_workspace(t, Float64, dims)
    @test_call target_modules = (Surfacelets,) make_workspace(t, x)
    ws = make_workspace(t, Float64, dims)
    @test_call target_modules = (Surfacelets,) analyze(t, x; workspace = ws)
    coeffs = analyze(t, x; workspace = ws)
    @test_call target_modules = (Surfacelets,) synthesize(t, coeffs; workspace = ws)
    @test_call target_modules = (Surfacelets,) synthesize!(dst, t, coeffs; workspace = ws)
    @test_call target_modules = (Surfacelets,) subband_layout(t, dims)

    # The decimated transform needs sizes its 3/2 pyramid can divide.
    s = Surfacelet(scales = 1, direction_levels = [1])
    sdims = (24, 24, 24)
    xs = randn(sdims...)
    @test_call target_modules = (Surfacelets,) make_workspace(s, Float64, sdims)
    sws = make_workspace(s, Float64, sdims)
    @test_call target_modules = (Surfacelets,) analyze(s, xs; workspace = sws)
    sc = analyze(s, xs; workspace = sws)
    @test_call target_modules = (Surfacelets,) synthesize(s, sc; workspace = sws)
    @test_call target_modules = (Surfacelets,) surfacelet_redundancy(s, sdims)
    @test_call target_modules = (Surfacelets,) subband_layout(s, sdims)

    @test_call target_modules = (Surfacelets,) synthesize!(similar(xs), s, sc; workspace = sws)
    @test_call target_modules = (Surfacelets,) analyze!(sc, s, xs; workspace = sws)
    zs = complex.(xs, xs)
    zws = make_workspace(s, zs)
    @test_call target_modules = (Surfacelets,) analyze(s, zs; workspace = zws)
end

@testitem "@test_opt Surfacelets core functions" tags = [:jet] begin
    using Surfacelets: make_hourglass, hourglass_decompose, hourglass_reconstruct,
        make_ndfb, ndfb_decompose
    using JET, Random
    Random.seed!(101)

    dims = (16, 16, 16)
    x = randn(dims...)
    dst = similar(x)

    # ── Directional FB ───────────────────────────────────────────────────────
    hg = make_hourglass(Float64, dims; bernstein_order = 6)
    @test_opt target_modules = (Surfacelets,) hourglass_decompose(x, hg)
    hs = hourglass_decompose(x, hg)
    @test_opt target_modules = (Surfacelets,) hourglass_reconstruct(hs, hg)

    fb = make_ndfb(Float64, dims; levels = 1, bernstein_order = 6)
    @test_opt target_modules = (Surfacelets,) ndfb_decompose(x, fb)

    # ── Transforms ───────────────────────────────────────────────────────────
    t = NSSurfacelet(scales = 1, direction_levels = [1], bernstein_order = 6)
    ws = make_workspace(t, Float64, dims)
    coeffs = analyze(t, x; workspace = ws)
    @test_opt target_modules = (Surfacelets,) analyze!(coeffs, t, x; workspace = ws)
    @test_opt target_modules = (Surfacelets,) synthesize!(dst, t, coeffs; workspace = ws)
end

@testitem "@test_call Surfacelets remaining API" tags = [:jet] begin
    using Surfacelets: HourglassFB, NDFB, make_hourglass, hourglass_mask, hourglass_weight,
        hourglass_decompose, hourglass_decompose!, hourglass_reconstruct!, make_ndfb,
        ndfb_sharing, ndfb_subband_index, ndfb_subband_triple, ndfb_mask, ndfb_weight,
        ndfb_decompose, ndfb_decompose!, ndfb_reconstruct!
    using JET, Random
    Random.seed!(102)
    M = (Surfacelets,)

    dims = (16, 16, 16)
    x = randn(dims...)

    # ── Filter tables ────────────────────────────────────────────────────────
    @test_call target_modules = M checkerboard_orders()

    # ── Hourglass bank ───────────────────────────────────────────────────────
    hg = make_hourglass(Float64, dims; bernstein_order = 6)
    @test hg isa HourglassFB
    @test_call target_modules = M hourglass_mask(hg, 1)
    @test_call target_modules = M hourglass_weight(hg, 2)
    hs = hourglass_decompose(x, hg)
    @test_call target_modules = M hourglass_decompose!(map(similar, hs), x, hg)
    @test_call target_modules = M hourglass_reconstruct!(similar(x), hs, hg)

    # ── Nonsubsampled directional bank ───────────────────────────────────────
    fb = make_ndfb(Float64, dims; levels = 1, bernstein_order = 6)
    @test fb isa NDFB
    @test_call target_modules = M ndfb_sharing(hg; levels = 1)
    @test_call target_modules = M ndfb_subband_index(fb, 1, 1, 1)
    @test_call target_modules = M ndfb_subband_triple(fb, 1)
    @test_call target_modules = M ndfb_mask(fb, 1)
    @test_call target_modules = M ndfb_weight(fb, 1)
    sbs = ndfb_decompose(x, fb)
    @test_call target_modules = M ndfb_decompose!([similar(s) for s in sbs], x, fb)
    @test_call target_modules = M ndfb_reconstruct!(similar(x), sbs, fb)

    # ── Transforms ───────────────────────────────────────────────────────────
    t = NSSurfacelet(scales = 1, direction_levels = [1], bernstein_order = 6)
    @test_call target_modules = M nsubbands(t, dims)
    ws = make_workspace(t, Float64, dims)
    @test ws isa NSSurfaceletWorkspace
    coeffs = analyze(t, x; workspace = ws)
    @test_call target_modules = M analyze!(coeffs, ws, x)
    @test_call target_modules = M analyze(t, x; threaded = false)
    @test_call target_modules = M synthesize(t, coeffs; workspace = ws, threaded = true)
    @test_call target_modules = M synthesize(ws, coeffs)
    @test_call target_modules = M subband(t, coeffs, 2)

    s = Surfacelet(scales = 1, direction_levels = [1])
    @test_call target_modules = M nsubbands(s, (24, 24, 24))
    sws = make_workspace(s, Float64, (24, 24, 24))
    @test sws isa SurfaceletWorkspace
    @test_call target_modules = M analyze(sws, randn(24, 24, 24))

    # ── Streaming (callback) forms ───────────────────────────────────────────
    @test_call target_modules = M ndfb_decompose((k, band) -> nothing, x, fb)
    @test_call target_modules = M ndfb_reconstruct!(k -> sbs[k], similar(x), fb)
    @test_call target_modules = M analyze((i, band) -> nothing, t, x; workspace = ws)
    @test_call target_modules = M synthesize!(i -> subband(t, coeffs, i), similar(x), t; workspace = ws)
end

@testitem "@test_opt Surfacelets remaining API" tags = [:jet] begin
    using Surfacelets: make_hourglass, hourglass_mask, hourglass_weight,
        hourglass_decompose, hourglass_decompose!, hourglass_reconstruct!, make_ndfb,
        ndfb_sharing, ndfb_nsubbands, ndfb_subband_index, ndfb_subband_triple, ndfb_mask,
        ndfb_weight, ndfb_decompose, ndfb_decompose!, ndfb_reconstruct, ndfb_reconstruct!
    using JET, Random
    Random.seed!(103)
    M = (Surfacelets,)

    dims = (16, 16, 16)
    x = randn(dims...)

    # ── Types / constructors / filters ───────────────────────────────────────
    @test_opt target_modules = M NSSurfacelet(scales = 2, direction_levels = [1, 1], bernstein_order = 6)
    @test_opt target_modules = M Surfacelet(scales = 2, direction_levels = [1, 1])
    @test_opt target_modules = M checkerboard_orders()

    # ── Hourglass bank ───────────────────────────────────────────────────────
    @test_opt target_modules = M make_hourglass(Float64, dims; bernstein_order = 6)
    hg = make_hourglass(Float64, dims; bernstein_order = 6)
    @test_opt target_modules = M hourglass_mask(hg, 1)
    @test_opt target_modules = M hourglass_weight(hg, 2)
    hs = hourglass_decompose(x, hg)
    @test_opt target_modules = M hourglass_decompose!(map(similar, hs), x, hg)
    @test_opt target_modules = M hourglass_reconstruct!(similar(x), hs, hg)

    # ── Nonsubsampled directional bank ───────────────────────────────────────
    @test_opt target_modules = M make_ndfb(Float64, dims; levels = 1, bernstein_order = 6)
    fb = make_ndfb(Float64, dims; levels = 1, bernstein_order = 6)
    @test_opt target_modules = M ndfb_sharing(hg; levels = 1)
    @test_opt target_modules = M ndfb_nsubbands(fb)
    @test_opt target_modules = M ndfb_subband_index(fb, 1, 1, 1)
    @test_opt target_modules = M ndfb_subband_triple(fb, 1)
    @test_opt target_modules = M ndfb_mask(fb, 1)
    @test_opt target_modules = M ndfb_weight(fb, 1)
    sbs = ndfb_decompose(x, fb)
    @test_opt target_modules = M ndfb_decompose!([similar(s) for s in sbs], x, fb)
    @test_opt target_modules = M ndfb_reconstruct(sbs, fb)
    @test_opt target_modules = M ndfb_reconstruct!(similar(x), sbs, fb)

    # ── Nonsubsampled surfacelet transform ───────────────────────────────────
    t = NSSurfacelet(scales = 1, direction_levels = [1], bernstein_order = 6)
    @test_opt target_modules = M nsubbands(t, dims)
    @test_opt target_modules = M make_workspace(t, Float64, dims)
    ws = make_workspace(t, Float64, dims)
    coeffs = analyze(t, x; workspace = ws)
    @test_opt target_modules = M analyze!(coeffs, t, x; workspace = ws)
    @test_opt target_modules = M synthesize(t, coeffs; workspace = ws)

    # ── Decimated surfacelet transform ───────────────────────────────────────
    s = Surfacelet(scales = 1, direction_levels = [1])
    sdims = (24, 24, 24)
    xs = randn(sdims...)
    @test_opt target_modules = M make_workspace(s, Float64, sdims)
    sws = make_workspace(s, Float64, sdims)
    @test_opt target_modules = M surfacelet_redundancy(s, sdims)
    @test_opt target_modules = M analyze(s, xs; workspace = sws)
    sc = analyze(s, xs; workspace = sws)
    @test_opt target_modules = M analyze!(sc, s, xs; workspace = sws)
    @test_opt target_modules = M synthesize(s, sc; workspace = sws)
    @test_opt target_modules = M synthesize!(similar(xs), s, sc; workspace = sws)

    # ── Streaming (callback) forms ───────────────────────────────────────────
    # Bind the captured variables with `let`: closures over top-level test variables return `Any`.
    by_index = let sbs = sbs
        k -> sbs[k]
    end
    by_subband = let coeffs = coeffs, t = t
        i -> subband(t, coeffs, i)
    end
    @test_opt target_modules = M ndfb_decompose((k, band) -> nothing, x, fb)
    @test_opt target_modules = M ndfb_reconstruct!(by_index, similar(x), fb)
    @test_opt target_modules = M analyze((i, band) -> nothing, t, x; workspace = ws)
    @test_opt target_modules = M synthesize!(by_subband, similar(x), t; workspace = ws)
end
