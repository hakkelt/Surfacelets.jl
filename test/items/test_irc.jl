# Golden-data regression of the decimated surfacelet transform against SurfBox,
# Lu and Do's reference implementation (MATLAB File Exchange 14485), stage by
# stage.  `test/reference/gen_surfbox_irc.m` writes the data; see its header for
# the format.  Large arrays are compared through fingerprints (two weighted
# sums and the energy), which catch any error above ~1e-13 of the norm.

@testmodule SurfBoxIRC begin
    export golden, testvol, fingerprint, fp_match

    const DIR = joinpath(@__DIR__, "..", "reference")

    # A fingerprint's length depends on whether the array is complex (5 numbers)
    # or real (3), which the index does not record, so the caller says which names
    # are complex.
    function _load_typed(complex_names)
        d = Dict{String, Any}()
        open(joinpath(DIR, "surfbox_irc.bin")) do io
            for ln in eachline(joinpath(DIR, "surfbox_irc.txt"))
                f = split(ln)
                name, kind, nd = f[1], f[2], parse(Int, f[3])
                sz = Tuple(parse.(Int, f[4:(3 + nd)]))
                if kind == "full"
                    d[name] = reshape(read!(io, Vector{Float64}(undef, prod(sz))), sz)
                else
                    n = complex_names(name) ? 5 : 3
                    d[name] = (size = sz, fp = read!(io, Vector{Float64}(undef, n)))
                end
            end
        end
        return d
    end

    # Spectra (hourglass, IRC and pyramid outputs) are complex; everything else is
    # a real spatial array.
    const golden = _load_typed(n -> startswith(n, "hg_") || startswith(n, "irc_") || startswith(n, "pyr_"))

    # Deterministic broadband test volume, exact in double precision (mirrors
    # `testvol` in the generator).
    function testvol(n)
        return [
            mod(73i^2 + 179j^2 + 283k^2 + 31i * j + 47j * k + 61k * i, 1031) / 1031 - 0.5
                for i in 1:n, j in 1:n, k in 1:n
        ]
    end

    function fingerprint(A::AbstractArray{T, 3}) where {T}
        s1 = zero(T)
        s2 = zero(T)
        for I in CartesianIndices(A)
            i, j, k = Tuple(I)
            s1 += A[I] * cos(0.37i + 0.91j + 1.53k)
            s2 += A[I] * sin(1.11i - 0.29j + 0.77k)
        end
        e = sum(abs2, A)
        return T <: Complex ? [real(s1), imag(s1), real(s2), imag(s2), e] : [s1, s2, e]
    end

    # A fingerprint matches when every entry agrees to `rtol` of the array's norm.
    function fp_match(A, ref; rtol = 1.0e-11)
        size(A) == ref.size || return false
        f = fingerprint(A)
        scale = sqrt(ref.fp[end])
        return all(abs.(f[1:(end - 1)] .- ref.fp[1:(end - 1)]) .<= rtol * scale) &&
            abs(f[end] - ref.fp[end]) <= rtol * ref.fp[end]
    end
end

@testitem "checkerboard filters reproduce SurfBox's" tags = [:directional, :surfbox] setup = [SurfBoxIRC] begin
    # The closed forms in `checkerboard.jl`, against the DTFT of the 2-D spatial
    # filters SurfBox ships, at the analysis and synthesis side.
    function dtft(H, c, w1, w2)
        return sum(H[i, j] * cis(-(w1 * (i - c[1]) + w2 * (j - c[2]))) for i in axes(H, 1), j in axes(H, 2))
    end
    ws = range(0, 2π; length = 13)[1:12]
    for bo in (4, 8, 12)
        t = Surfacelets._cbd_ladder(bo)
        worst = 0.0
        for a in ws, b in ws, ch in 0:1
            tag = ch == 0 ? "H0" : "H1"
            ctag = ch == 0 ? "c0" : "c1"
            hd = dtft(golden["cbd_$(bo)_dec_$(tag)"], golden["cbd_$(bo)_dec_$(ctag)"], a, b)
            hr = dtft(golden["cbd_$(bo)_rec_$(tag)"], golden["cbd_$(bo)_rec_$(ctag)"], a, b)
            worst = max(
                worst, abs(Surfacelets._cbd_analysis(t, ch, a, b) - hd),
                abs(Surfacelets._cbd_synthesis(t, ch, a, b) - hr)
            )
        end
        @test worst < 1.0e-13
    end
end

@testitem "ritf hourglass and IRC stages reproduce SurfBox" tags = [:directional, :surfbox] setup = [SurfBoxIRC] begin
    S = Surfacelets
    x = testvol(32)
    X = S.fft(complex.(x))
    V = S._ritf_masks(Float64, size(x))
    for K in 1:3
        half = ntuple(d -> d == K ? (1:17) : (1:32), 3)
        hg = X[half...] .* V[K]
        @test fp_match(hg, golden["hg_$K"]; rtol = 1.0e-10)
        for (nm, L) in (("l111", [-1 1 1; 1 -1 1; 1 1 -1]), ("l212", [-1 2 1; 1 -1 2; 2 1 -1]))
            levels = ntuple(d -> d == K ? 0 : L[K, d], 3)
            st = S._irc_stages(Float64, size(hg), 32, K, levels, 4)
            Sk = copy(hg)
            tmp = similar(Sk, length(Sk) ÷ 2)
            S._irc_analysis!(Sk, st, tmp)
            @test fp_match(Sk, golden["irc_$(nm)_$K"]; rtol = 1.0e-10)
            S._irc_synthesis!(Sk, st, tmp)
            @test maximum(abs, Sk .- hg) < 1.0e-12 * maximum(abs, hg)
        end
    end
    # Σ V_K² = 3 on each half-spectrum: a tight frame of redundancy 3.
    Vf = S._ritf_masks(Float64, (16, 20, 12))
    @test maximum(abs, Vf[1][1:9, 1:11, 1:7] .^ 2 .+ Vf[2][1:9, 1:11, 1:7] .^ 2 .+ Vf[3][1:9, 1:11, 1:7] .^ 2 .- 3) < 1.0e-12
end

@testitem "NDFB subbands reproduce SurfBox" tags = [:directional, :surfbox] setup = [SurfBoxIRC] begin
    S = Surfacelets
    x = testvol(32)
    for (nm, L) in (("l111", [-1 1 1; 1 -1 1; 1 1 -1]), ("l212", [-1 2 1; 1 -1 2; 2 1 -1]))
        lv = S._surf_levels(L)
        masks = S._ritf_masks(Float64, size(x))
        stages = ntuple(K -> S._irc_stages(Float64, ntuple(d -> d == K ? 17 : 32, 3), 32, K, lv[K], 4), 3)
        sc = S._SurfScale(size(x), lv, masks, stages)
        bands = S._surf_scale_forward(S.fft(complex.(x)), sc)
        i = 0
        for K in 1:3, w in 1:(1 << sum(lv[K]))
            i += 1
            @test fp_match(bands[i], golden["ndfb_$(nm)_$(K)_$(w)"])
        end
        rec = real.(S.ifft(S._surf_scale_inverse(bands, sc)))
        @test fp_match(rec, golden["ndfbrec_$(nm)"])
        @test maximum(abs, rec .- x) < 1.0e-13
    end
end

@testitem "Fourier pyramid reproduces SurfBox" tags = [:pyramid, :surfbox] setup = [SurfBoxIRC] begin
    S = Surfacelets
    for (tag, n, mode) in (("m15", 48, 1.5), ("m1", 32, 1), ("m2", 32, 2))
        x = testvol(n)
        levels = S._fourier_pyramid(Float64, size(x), 2, mode)
        X = S.fft(complex.(x))
        orig = copy(X)
        highs = Any[]
        for lv in levels
            low = S._pyramid_split!(X, lv)
            push!(highs, X)
            X = low
        end
        @test fp_match(highs[1], golden["pyr_$(tag)_1"])
        @test fp_match(highs[2], golden["pyr_$(tag)_2"])
        @test fp_match(X, golden["pyr_$(tag)_3"])
        for k in 2:-1:1
            X = S._pyramid_merge!(highs[k], X, levels[k])
        end
        @test maximum(abs, X .- orig) < 1.0e-12 * maximum(abs, orig)
    end
end

@testitem "surfacelet transform reproduces SurfBox's Surfdec / Surfrec" tags = [:surfacelet, :surfbox] setup = [SurfBoxIRC] begin
    L111 = [-1 1 1; 1 -1 1; 1 1 -1]
    L212 = [-1 2 1; 1 -1 2; 2 1 -1]
    for (tag, n, mode, Ls, bo) in (("m15", 48, 1.5, [L111, L111], 8), ("m2", 64, 2, [L212, L111], 4))
        x = testvol(n)
        p = SurfaceletParams(J = 2, L_array = Ls, pyr_mode = mode, bo = bo)
        c = surfacelet_forward(x, p)
        for j in 1:2
            i = 0
            for K in 1:3, w in 1:(1 << sum(p.levels[j][K]))
                i += 1
                @test fp_match(c.subbands[j][i], golden["surf_$(tag)_$(j)_$(K)_$(w)"])
            end
        end
        @test fp_match(c.coarse, golden["surf_$(tag)_low"])
        r = surfacelet_inverse(c, p)
        @test fp_match(r, golden["surfrec_$(tag)"])
        @test maximum(abs, r .- x) < 1.0e-13
    end
end

@testitem "surfacelet redundancy matches SurfBox" tags = [:surfacelet, :surfbox] setup = [SurfBoxIRC] begin
    # SurfBox at 96³, pyramid mode 1.5: 24 directions per scale over three
    # scales, and 96 over two — about 4 either way, flat in the direction count.
    for (nm, Lm, J) in (("d24", [-1 1 2; 2 -1 1; 1 2 -1], 3), ("d96", [-1 2 3; 3 -1 2; 2 3 -1], 2))
        pl = make_surfacelet(SurfaceletParams(J = J, L_array = fill(Lm, J), bo = 4), (96, 96, 96))
        @test surfacelet_redundancy(pl) ≈ golden["red_96_$(nm)"][1]
    end
end
