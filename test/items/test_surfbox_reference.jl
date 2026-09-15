# Golden-data regression against SurfBox, the reference implementation of the
# surfacelet transform (Yue Lu and Minh N. Do, MATLAB File Exchange 14485).
#
# `test/reference/surfbox_hourglass_32.bin` holds the first hourglass analysis
# mask of SurfBox's `HourGlassDec` on a 32^3 grid, restricted to the
# non-negative-frequency octant. `test/reference/gen_surfbox_golden.m`
# regenerates it. SurfBox normalises pointwise so that `sum_k V_k^2 == 3`; this
# package instead builds a partition of unity, so the comparable quantity is
# `W_k == V_k^2 / 3`.
#
# The two constructions are different discrete realisations of the same ideal
# hourglass partition -- SurfBox raises the magnitude response of a
# Kaiser-windowed diamond kernel to the power lambda, this package composes
# Bernstein mapping polynomials -- so the masks are compared on geometry
# (which hourglass dominates a given frequency) rather than value.

@testitem "hourglass matches the SurfBox reference partition" tags = [:directional] begin
    reference_octant() = reshape(
        reinterpret(Float32, read(joinpath(@__DIR__, "..", "reference", "surfbox_hourglass_32.bin"))),
        17, 17, 17
    )

    # Expand the octant to the full 32^3 grid: the mask is even along every axis.
    function expand_even(octant::AbstractArray{Float32, 3})
        n = 32
        mirror = [i <= 17 ? i : n - i + 2 for i in 1:n]
        return Float64[octant[mirror[i], mirror[j], mirror[k]] for i in 1:n, j in 1:n, k in 1:n]
    end

    v1 = expand_even(reference_octant())
    # Masks 2 and 3 are axis permutations of mask 1.
    v = (v1, permutedims(v1, (2, 1, 3)), permutedims(v1, (3, 2, 1)))

    # SurfBox is a tight frame with redundancy 3.
    tightness = v[1] .^ 2 .+ v[2] .^ 2 .+ v[3] .^ 2
    @test all(x -> isapprox(x, 3.0; atol = 1.0e-5), tightness)

    # Restrict to the rfft grid this package works on, and convert to weights.
    w_ref = ntuple(k -> (v[k][1:17, :, :] .^ 2) ./ 3, 3)

    fb = make_hourglass((32, 32, 32), Float64)
    w_mine = ntuple(k -> hourglass_weight(fb, k), 3)

    @test all(size(w_mine[k]) == size(w_ref[k]) for k in 1:3)

    # Same axis convention: each mask matches its own reference far better than
    # it matches either of the others.
    for k in 1:3, m in 1:3
        d = maximum(abs, w_mine[k] .- w_ref[m])
        k == m ? (@test d < 0.5) : (@test d > 0.9)
    end

    # Geometry: the dominant hourglass agrees at almost every frequency, and
    # every disagreement is a near-tie in the reference.
    dominant_mine(i) = argmax((w_mine[1][i], w_mine[2][i], w_mine[3][i]))
    dominant_ref(i) = argmax((w_ref[1][i], w_ref[2][i], w_ref[3][i]))
    function ref_gap(i)
        sorted = sort([w_ref[1][i], w_ref[2][i], w_ref[3][i]]; rev = true)
        return sorted[1] - sorted[2]
    end
    disagree = [i for i in eachindex(w_mine[1]) if dominant_mine(i) != dominant_ref(i)]
    @test 1 - length(disagree) / length(w_mine[1]) > 0.94
    @test maximum(ref_gap, disagree) < 0.5

    # Values track the reference closely away from the transition bands.
    mean_abs = sum(k -> sum(abs, w_mine[k] .- w_ref[k]), 1:3) / (3 * length(w_mine[1]))
    @test mean_abs < 0.04

    # The weights are a partition of unity; the analysis masks are their square
    # roots, so this package is tight the same way SurfBox is. The two differ
    # only in the constant: SurfBox normalises to `sum_k V_k^2 == N`, this
    # package to 1, so `V_surfbox == sqrt(N) * V_mine` up to the transition-band
    # differences measured above.
    unity = w_mine[1] .+ w_mine[2] .+ w_mine[3]
    @test all(x -> isapprox(x, 1.0; atol = 1.0e-12), unity)
    masks = ntuple(k -> hourglass_mask(fb, k), 3)
    parseval = masks[1] .^ 2 .+ masks[2] .^ 2 .+ masks[3] .^ 2
    @test all(x -> isapprox(x, 1.0; atol = 1.0e-12), parseval)

    # Without the normalisation the masks are the weights themselves, and the
    # frame-bound ratio is 3 rather than 1.
    loose = make_hourglass((32, 32, 32), Float64; tight = false)
    energy = sum(hourglass_mask(loose, k) .^ 2 for k in 1:3)
    @test isapprox(minimum(energy), 1 / 3; atol = 1.0e-3)
    @test isapprox(maximum(energy), 1.0; atol = 1.0e-12)
end

# `test/reference/surfbox_directions_32.bin` records which of SurfBox's 12
# directional subbands carries the most energy for each of 80 plane waves.
# SurfBox's NDFB is decimated and built from an iteratively resampled
# checkerboard chain, this package's `ndfb` places the same ideal wedge
# boundaries directly in frequency, and neither orders its subbands the same
# way -- so the test asks whether the two agree on the *partition* of
# directions, up to relabelling.

@testitem "ndfb matches the SurfBox reference wedge partition" tags = [:directional] begin
    raw = reshape(
        reinterpret(Float64, read(joinpath(@__DIR__, "..", "reference", "surfbox_directions_32.bin"))),
        :
    )
    ndirs = 80
    dirs = reshape(raw[1:(3 * ndirs)], 3, ndirs)
    ref_label = Int.(raw[(3 * ndirs + 1):end])
    @test length(ref_label) == ndirs
    @test length(unique(ref_label)) == 12

    fb = make_ndfb((32, 32, 32), Float64; levels = 2)
    @test ndfb_nsubbands(fb) == 12

    energies = zeros(ndirs, 12)
    for t in 1:ndirs
        k = round.(Int, dirs[:, t] .* (0.6 * 16))
        all(iszero, k) && (k = [1, 0, 0])
        vol = [
            cospi(2 * (k[1] * (i - 1) + k[2] * (j - 1) + k[3] * (l - 1)) / 32)
                for i in 1:32, j in 1:32, l in 1:32
        ]
        energies[t, :] = [sum(abs2, b) for b in ndfb_decompose(vol, fb)]
    end
    my_label = [argmax(@view energies[t, :]) for t in 1:ndirs]
    @test length(unique(my_label)) == 12

    # Same partition: two directions share a subband here exactly when they
    # share one in SurfBox.
    same = count(
        (ref_label[a] == ref_label[b]) == (my_label[a] == my_label[b])
            for a in 1:ndirs, b in 1:ndirs
    )
    @test same / ndirs^2 > 0.93

    # The confusion matrix induces a bijection between the two labellings.
    confusion = zeros(Int, 12, 12)
    for t in 1:ndirs
        confusion[my_label[t], ref_label[t]] += 1
    end
    relabel = [argmax(@view confusion[m, :]) for m in 1:12]
    @test length(unique(relabel)) == 12
    @test count(t -> relabel[my_label[t]] == ref_label[t], 1:ndirs) / ndirs > 0.8

    # Every direction the two disagree on is a runner-up here, never further
    # down -- the mismatches sit on wedge boundaries, not in the wrong wedge.
    inverse = zeros(Int, 12)
    for m in 1:12
        inverse[relabel[m]] = m
    end
    for t in 1:ndirs
        top2 = partialsortperm(@view(energies[t, :]), 1:2; rev = true)
        @test inverse[ref_label[t]] in top2
    end
end
