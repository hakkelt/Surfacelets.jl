@testitem "Generic axes — transforms" tags = [:quality] begin
    using OffsetArrays: OffsetArray
    using MultiscaleTransformsBase: analyze, analyze!, synthesize, synthesize!, similar_coefficients, subband
    using Random

    offset(a) = OffsetArray(a, (ntuple(_ -> 2, ndims(a))))
    msg = "offset arrays are not supported"

    # A view of a larger array is indexed from 1 and must give the plain array's result.
    function embedded(x)
        big = zeros(eltype(x), size(x) .+ 3)
        big[axes(x)...] .= x
        return view(big, axes(x)...)
    end

    @testset "NSSurfacelet" begin
        x = randn(Xoshiro(1), 16, 16, 16)
        xv = embedded(x)
        t = NSSurfacelet(scales = 1, direction_levels = [1], bernstein_order = 4)
        c = analyze(t, x)
        @test analyze(t, xv) == c
        @test synthesize(t, c) ≈ x
        cv = embedded(c)
        @test synthesize(t, copy(cv)) == synthesize(t, c)
        @test synthesize!(similar(x), t, cv) ≈ synthesize(t, c)
        @test_throws msg analyze(t, offset(x))
        @test_throws msg analyze!(similar(c), t, offset(x))
        @test_throws msg analyze!(offset(similar(c)), t, x)
        @test_throws msg synthesize!(offset(similar(x)), t, c)
        @test_throws msg synthesize!(similar(x), t, offset(c))
        @test_throws msg analyze((i, b) -> nothing, t, offset(x))
        @test_throws msg synthesize!(i -> view(c, :, :, :, i), offset(similar(x)), t)
        @test_throws msg synthesize!(i -> offset(c[:, :, :, i]), similar(x), t)
        z = complex.(x, x)
        @test analyze(t, embedded(z)) == analyze(t, z)
        @test_throws msg analyze(t, offset(z))
    end

    @testset "Surfacelet" begin
        x = randn(Xoshiro(2), 24, 24, 24)
        xv = embedded(x)
        t = Surfacelet(scales = 1, direction_levels = [1], checkerboard_order = 6)
        c = analyze(t, x)
        @test Vector(analyze(t, xv)) == Vector(c)
        @test synthesize!(similar(x), t, c) == synthesize(t, c)
        @test synthesize!(embedded(similar(x)), t, c) == synthesize(t, c)
        @test_throws msg analyze(t, offset(x))
        @test_throws msg analyze!(similar_coefficients(t, x), t, offset(x))
        @test_throws msg synthesize!(offset(similar(x)), t, c)
        z = complex.(x, x)
        @test_throws msg analyze(t, offset(z))
    end
end

@testitem "Generic axes — directional banks and filters" tags = [:quality] begin
    using OffsetArrays: OffsetArray
    using Surfacelets: make_hourglass, hourglass_decompose, hourglass_decompose!, hourglass_reconstruct,
        hourglass_reconstruct!, make_ndfb, ndfb_nsubbands, ndfb_decompose, ndfb_decompose!, ndfb_reconstruct,
        ndfb_reconstruct!, mapping_coefficients, mapping_response, bernstein_table, dft_frequencies
    using Random

    offset(a) = OffsetArray(a, (ntuple(_ -> 2, ndims(a))))
    msg = "offset arrays are not supported"
    function embedded(x)
        big = zeros(eltype(x), size(x) .+ 3)
        big[axes(x)...] .= x
        return view(big, axes(x)...)
    end

    x = randn(Xoshiro(3), 12, 12, 12)
    z = complex.(x, reverse(x))

    @testset "hourglass" begin
        fb = make_hourglass(Float64, size(x); bernstein_order = 4)
        for v in (x, z)
            s = hourglass_decompose(v, fb)
            @test hourglass_decompose(embedded(v), fb) == s
            @test hourglass_reconstruct(embedded.(s), fb) == hourglass_reconstruct(s, fb)
            @test_throws msg hourglass_decompose(offset(v), fb)
            @test_throws msg hourglass_decompose!(offset.(s), v, fb)
            @test_throws msg hourglass_reconstruct(offset.(s), fb)
            @test_throws msg hourglass_reconstruct!(offset(similar(v)), s, fb)
        end
        fbl = make_hourglass(Float64, size(x); bernstein_order = 4, tight = false)
        s = hourglass_decompose(x, fbl)
        @test_throws msg hourglass_reconstruct!(similar(x), offset.(s), fbl)
    end

    @testset "ndfb" begin
        fb = make_ndfb(Float64, size(x); levels = 1, bernstein_order = 4)
        n = ndfb_nsubbands(fb)
        for v in (x, z)
            s = ndfb_decompose(v, fb)
            @test ndfb_decompose(embedded(v), fb) == s
            @test ndfb_reconstruct(embedded.(s), fb) == ndfb_reconstruct(s, fb)
            @test_throws msg ndfb_decompose(offset(v), fb)
            @test_throws msg ndfb_decompose!(similar(s), offset(v), fb)
            @test_throws msg ndfb_decompose!(offset(similar(s)), v, fb)
            @test_throws msg ndfb_decompose!([similar(s[1]), offset(similar(s[1])), s[3:n]...], v, fb)
            @test_throws msg ndfb_decompose((i, b) -> nothing, offset(v), fb)
            @test_throws msg ndfb_reconstruct!(offset(similar(v)), s, fb)
            @test_throws msg ndfb_reconstruct!(similar(v), offset(s), fb)
            @test_throws msg ndfb_reconstruct!(similar(v), [s[1], offset(s[2]), s[3:n]...], fb)
            @test_throws msg ndfb_reconstruct!(k -> offset(s[k]), similar(v), fb)
            @test_throws msg ndfb_reconstruct!(k -> s[k], offset(similar(v)), fb)
        end
    end

    @testset "levels matrix" begin
        L = [-1 2 0; 1 -1 1; 0 3 -1]
        @test Surfacelet(scales = 1, direction_levels = [embedded(L)]) == Surfacelet(scales = 1, direction_levels = [L])
        @test_throws msg Surfacelet(scales = 1, direction_levels = [offset(L)])
    end

    @testset "mapping_response" begin
        N = 4
        ω = dft_frequencies(8)
        tbl = bernstein_table(N, ω)
        c1 = mapping_coefficients(N, 2, (1,))
        @test mapping_response(embedded(c1), (embedded(tbl), embedded(tbl))) == mapping_response(c1, (tbl, tbl))
        @test_throws msg mapping_response(offset(c1), (tbl, tbl))
        @test_throws msg mapping_response(c1, (offset(tbl), tbl))
        c3 = mapping_coefficients(N, 3, (1,))
        @test mapping_response(embedded(c3), (tbl, tbl, tbl)) == mapping_response(c3, (tbl, tbl, tbl))
        @test_throws msg mapping_response(offset(c3), (tbl, tbl, tbl))
        @test_throws msg mapping_response(c3, (tbl, tbl, offset(tbl)))
        @test_throws msg mapping_response(offset(c1[:, 1]), (tbl,))
    end
end
