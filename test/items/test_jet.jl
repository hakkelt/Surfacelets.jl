@testitem "@test_call Surfacelets public API" tags = [:jet] begin
    using JET, Random
    Random.seed!(100)

    dims = (16, 16, 16)
    x = randn(dims...)
    dst = similar(x)

    # ── Types / constructors ─────────────────────────────────────────────────
    @test_call target_modules = (Surfacelets,) NSSurfaceletParams(J = 2, L_array = [1, 1], order = 6)
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
    p = NSSurfaceletParams(J = 1, L_array = [1], order = 6)
    @test_call target_modules = (Surfacelets,) make_ns_surfacelet_workspace(p, dims, Float64)
    ws = make_ns_surfacelet_workspace(p, dims, Float64)
    @test_call target_modules = (Surfacelets,) ns_surfacelet_forward(x, p; workspace = ws)
    coeffs = ns_surfacelet_forward(x, p; workspace = ws)
    @test_call target_modules = (Surfacelets,) ns_surfacelet_inverse(coeffs, p; workspace = ws)
    @test_call target_modules = (Surfacelets,) ns_surfacelet_inverse!(dst, coeffs, p; workspace = ws)

    # The decimated transform needs sizes its 3/2 pyramid can divide.
    sp = SurfaceletParams(J = 1, L_array = [1])
    sdims = (24, 24, 24)
    xs = randn(sdims...)
    @test_call target_modules = (Surfacelets,) make_surfacelet(sp, sdims, Float64)
    spl = make_surfacelet(sp, sdims, Float64)
    @test_call target_modules = (Surfacelets,) surfacelet_forward(xs, sp; plan = spl)
    sc = surfacelet_forward(xs, sp; plan = spl)
    @test_call target_modules = (Surfacelets,) surfacelet_inverse(sc, sp; plan = spl)
    @test_call target_modules = (Surfacelets,) surfacelet_redundancy(spl)

    hp = STISurfaceletParams(J = 2, L_array = [1, 1])
    @test_call target_modules = (Surfacelets,) make_sti_surfacelet(hp, dims, Float64)
    hpl = make_sti_surfacelet(hp, dims, Float64)
    @test_call target_modules = (Surfacelets,) sti_surfacelet_forward(x, hp; plan = hpl)
    hc = sti_surfacelet_forward(x, hp; plan = hpl)
    @test_call target_modules = (Surfacelets,) sti_surfacelet_inverse(hc, hp; plan = hpl)
    @test_call target_modules = (Surfacelets,) sti_surfacelet_inverse!(similar(x), hc, hp; plan = hpl)
    @test_call target_modules = (Surfacelets,) sti_surfacelet_forward!(hc, x, hp; plan = hpl)
    @test_call target_modules = (Surfacelets,) surfacelet_inverse!(similar(xs), sc, sp; plan = spl)
    @test_call target_modules = (Surfacelets,) surfacelet_forward!(sc, xs, sp; plan = spl)
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
    p = NSSurfaceletParams(J = 1, L_array = [1], order = 6)
    ws = make_ns_surfacelet_workspace(p, dims, Float64)
    @test_opt target_modules = (Surfacelets,) ns_surfacelet_forward(x, p; workspace = ws)
    coeffs = ns_surfacelet_forward(x, p; workspace = ws)
    @test_opt target_modules = (Surfacelets,) ns_surfacelet_inverse!(dst, coeffs, p; workspace = ws)
end

@testitem "@test_call Surfacelets remaining API" tags = [:jet] begin
    using JET, Random
    Random.seed!(102)
    M = (Surfacelets,)

    dims = (16, 16, 16)
    x = randn(dims...)
    h = [0.25, 0.5, 0.25]

    # ── Re-exported filter types ───────────────────────────────────────────────
    @test_call target_modules = M FilterPair(CDF97.h, CDF97.g)
    @test_call target_modules = M upsample_filter(CDF97.h, 2)
    @test_call target_modules = M checkerboard_orders()

    # ── Primitives ───────────────────────────────────────────────────────────
    @test_call target_modules = M conv3d_sep_fft(x, h, h, h)
    plan = make_conv3d_fft_plan(dims, Float64)
    @test plan isa Conv3DFFTPlan
    xc = complex.(x)
    @test_call target_modules = M conv3d_sep_fft!(similar(xc), xc, h, h, h, make_conv3d_fft_plan(dims, ComplexF64))

    # ── Pyramid (in place) ───────────────────────────────────────────────────
    c, bp = nsp3d_decompose(x, CDF97, 1)
    @test_call target_modules = M nsp3d_decompose!(similar(c), similar(bp), x, CDF97, 1)
    @test_call target_modules = M nsp3d_reconstruct!(similar(x), c, bp, CDF97, 1)

    # ── Hourglass bank ───────────────────────────────────────────────────────
    hg = make_hourglass(dims, Float64; order = 6)
    @test hg isa HourglassFB
    @test_call target_modules = M hourglass_mask(hg, 1)
    @test_call target_modules = M hourglass_weight(hg, 2)
    hs = hourglass_decompose(x, hg)
    @test_call target_modules = M hourglass_decompose!(map(similar, hs), x, hg)
    @test_call target_modules = M hourglass_reconstruct!(similar(x), hs, hg)

    # ── Nonsubsampled directional bank ───────────────────────────────────────
    fb = make_ndfb(dims, Float64; levels = 1, order = 6)
    @test fb isa NDFB
    @test_call target_modules = M ndfb_sharing(hg, 1)
    @test_call target_modules = M ndfb_subband_index(fb, 1, 1, 1)
    @test_call target_modules = M ndfb_subband_triple(fb, 1)
    @test_call target_modules = M ndfb_mask(fb, 1)
    @test_call target_modules = M ndfb_weight(fb, 1)
    sbs = ndfb_decompose(x, fb)
    @test_call target_modules = M ndfb_decompose!([similar(s) for s in sbs], x, fb)
    @test_call target_modules = M ndfb_reconstruct!(similar(x), sbs, fb)

    # ── Transforms ───────────────────────────────────────────────────────────
    p = NSSurfaceletParams(J = 1, L_array = [1], order = 6)
    @test_call target_modules = M ns_surfacelet_nsubbands(p, 1)
    ws = make_ns_surfacelet_workspace(p, dims, Float64)
    @test ws isa NSSurfaceletWorkspace
    coeffs = ns_surfacelet_forward(x, p; workspace = ws)
    @test coeffs isa NSSurfaceletCoefficients
    @test_call target_modules = M ns_surfacelet_forward!(coeffs, x, p; workspace = ws)
    @test_call target_modules = M ns_surfacelet_forward(x, p; threaded = false)
    @test_call target_modules = M ns_surfacelet_inverse(coeffs, p; workspace = ws, threaded = true)

    sp = SurfaceletParams(J = 1, L_array = [1])
    @test_call target_modules = M surfacelet_nsubbands(sp, 1)
    spl = make_surfacelet(sp, (24, 24, 24), Float64)
    @test spl isa SurfaceletPlan
    @test surfacelet_forward(randn(24, 24, 24), sp; plan = spl) isa SurfaceletCoefficients
    hp = STISurfaceletParams(J = 2, L_array = [1, 1])
    @test_call target_modules = M sti_surfacelet_nsubbands(hp, 1)

    # ── Streaming (callback) forms ───────────────────────────────────────────
    @test_call target_modules = M ndfb_decompose((k, band) -> nothing, x, fb)
    @test_call target_modules = M ndfb_reconstruct!(k -> sbs[k], similar(x), fb)
    @test_call target_modules = M ns_surfacelet_forward((j, k, band) -> nothing, x, p; workspace = ws)
    @test_call target_modules = M ns_surfacelet_inverse((j, k) -> coeffs.subbands[j][k], coeffs.coarse, p; workspace = ws)
    @test_call target_modules = M ns_surfacelet_inverse!(
        (j, k) -> coeffs.subbands[j][k], similar(x), coeffs.coarse, p; workspace = ws
    )
end

@testitem "@test_opt Surfacelets remaining API" tags = [:jet] begin
    using JET, Random
    Random.seed!(103)
    M = (Surfacelets,)

    dims = (16, 16, 16)
    x = randn(dims...)
    dst = similar(x)
    h = [0.25, 0.5, 0.25]

    # ── Types / constructors / filters ───────────────────────────────────────
    @test_opt target_modules = M NSSurfaceletParams(J = 2, L_array = [1, 1], order = 6)
    @test_opt target_modules = M SurfaceletParams(J = 2, L_array = [1, 1])
    @test_opt target_modules = M STISurfaceletParams(J = 2, L_array = [1, 1])
    @test_opt target_modules = M FilterPair(CDF97.h, CDF97.g)
    @test_opt target_modules = M upsample_filter(CDF97.h, 2)
    @test_opt target_modules = M checkerboard_orders()

    # ── Primitives ───────────────────────────────────────────────────────────
    @test_opt target_modules = M make_conv3d_fft_plan(dims, Float64)
    plan = make_conv3d_fft_plan(dims, Float64)
    @test_opt target_modules = M conv3d_sep_fft(x, h, h, h)
    @test_opt target_modules = M conv3d_sep_fft!(dst, x, h, h, h, plan)

    # ── Pyramid ──────────────────────────────────────────────────────────────
    c, bp = nsp3d_decompose(x, CDF97, 1)
    @test_opt target_modules = M nsp3d_decompose!(similar(c), similar(bp), x, CDF97, 1)
    @test_opt target_modules = M nsp3d_reconstruct!(similar(x), c, bp, CDF97, 1)

    # ── Hourglass bank ───────────────────────────────────────────────────────
    @test_opt target_modules = M make_hourglass(dims, Float64; order = 6)
    hg = make_hourglass(dims, Float64; order = 6)
    @test_opt target_modules = M hourglass_mask(hg, 1)
    @test_opt target_modules = M hourglass_weight(hg, 2)
    hs = hourglass_decompose(x, hg)
    @test_opt target_modules = M hourglass_decompose!(map(similar, hs), x, hg)
    @test_opt target_modules = M hourglass_reconstruct!(similar(x), hs, hg)

    # ── Nonsubsampled directional bank ───────────────────────────────────────
    @test_opt target_modules = M make_ndfb(dims, Float64; levels = 1, order = 6)
    fb = make_ndfb(dims, Float64; levels = 1, order = 6)
    @test_opt target_modules = M ndfb_sharing(hg, 1)
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
    p = NSSurfaceletParams(J = 1, L_array = [1], order = 6)
    @test_opt target_modules = M ns_surfacelet_nsubbands(p, 1)
    @test_opt target_modules = M make_ns_surfacelet_workspace(p, dims, Float64)
    ws = make_ns_surfacelet_workspace(p, dims, Float64)
    coeffs = ns_surfacelet_forward(x, p; workspace = ws)
    @test_opt target_modules = M ns_surfacelet_forward!(coeffs, x, p; workspace = ws)
    @test_opt target_modules = M ns_surfacelet_inverse(coeffs, p; workspace = ws)

    # ── Decimated surfacelet transform ───────────────────────────────────────
    sp = SurfaceletParams(J = 1, L_array = [1])
    sdims = (24, 24, 24)
    xs = randn(sdims...)
    @test_opt target_modules = M surfacelet_nsubbands(sp, 1)
    @test_opt target_modules = M make_surfacelet(sp, sdims, Float64)
    spl = make_surfacelet(sp, sdims, Float64)
    @test_opt target_modules = M surfacelet_redundancy(spl)
    @test_opt target_modules = M surfacelet_forward(xs, sp; plan = spl)
    sc = surfacelet_forward(xs, sp; plan = spl)
    @test_opt target_modules = M surfacelet_forward!(sc, xs, sp; plan = spl)
    @test_opt target_modules = M surfacelet_inverse(sc, sp; plan = spl)
    @test_opt target_modules = M surfacelet_inverse!(similar(xs), sc, sp; plan = spl)

    # ── Decimated shift-invariant surfacelet transform ───────────────────────
    hp = STISurfaceletParams(J = 2, L_array = [1, 1])
    @test_opt target_modules = M sti_surfacelet_nsubbands(hp, 1)
    @test_opt target_modules = M make_sti_surfacelet(hp, dims, Float64)
    hpl = make_sti_surfacelet(hp, dims, Float64)
    @test_opt target_modules = M sti_surfacelet_forward(x, hp; plan = hpl)
    hc = sti_surfacelet_forward(x, hp; plan = hpl)
    @test_opt target_modules = M sti_surfacelet_forward!(hc, x, hp; plan = hpl)
    @test_opt target_modules = M sti_surfacelet_inverse(hc, hp; plan = hpl)
    @test_opt target_modules = M sti_surfacelet_inverse!(similar(x), hc, hp; plan = hpl)

    # ── Streaming (callback) forms ───────────────────────────────────────────
    # Bind the captured variables with `let`: closures over top-level test variables return `Any`.
    by_index = let sbs = sbs
        k -> sbs[k]
    end
    by_scale = let coeffs = coeffs
        (j, k) -> coeffs.subbands[j][k]
    end
    @test_opt target_modules = M ndfb_decompose((k, band) -> nothing, x, fb)
    @test_opt target_modules = M ndfb_reconstruct!(by_index, similar(x), fb)
    @test_opt target_modules = M ns_surfacelet_forward((j, k, band) -> nothing, x, p; workspace = ws)
    @test_opt target_modules = M ns_surfacelet_inverse(by_scale, coeffs.coarse, p; workspace = ws)
    @test_opt target_modules = M ns_surfacelet_inverse!(by_scale, similar(x), coeffs.coarse, p; workspace = ws)
end
