"""
    Surfacelets

Three-dimensional nonsubsampled and decimated surfacelet transforms, as
transform objects of [MultiscaleTransformsBase](https://github.com/hakkelt/MultiscaleTransformsBase.jl),
whose interface this package re-exports.

## Quick reference

| Name | Description |
|------|-------------|
| [`NSSurfacelet`](@ref) | Nonsubsampled surfacelet transform: every subband has the size of the volume; streams |
| [`Surfacelet`](@ref) | Decimated counterpart, subbands at reduced resolution |
| `analyze` / `synthesize` (and `!` forms) | Run a transform |
| `make_workspace` | Masks, buffers and FFT plans for repeated calls on one size |
| `subband_layout` / `nsubbands` / `subband` | What each subband is, and where it is in the coefficients |
| `Surfacelets.make_hourglass` / `Surfacelets.make_ndfb` | The directional stage on its own (public, not exported) |

The surfacelet transform is a construction in its own right, not a
dimension-raised contourlet transform.  Its first directional stage carves the
spectrum into three hourglass-shaped regions around the three axes, a partition
with no quincunx-lattice description that Lu & Do prove cannot be critically
sampled.  The nonsubsampled transform builds it from Bernstein mapping kernels
and cuts each hourglass into `2^L` wedges with directly placed boundaries.

[`Surfacelet`](@ref) is Lu & Do's decimated surfacelet transform, reproducing
their reference toolbox SurfBox to rounding error: a 3/2-resampling Fourier
pyramid, SurfBox's hourglass bank, and iterated resampled checkerboard filter
banks that refine each hourglass into wedges while decimating critically.
About 4 coefficients per voxel, whatever the number of directions
([`surfacelet_redundancy`](@ref)); not shift-invariant.  See the "Theory" page.
"""
module Surfacelets

using Compat: @compat
using FFTW: FFTW, plan_rfft, plan_irfft, plan_brfft, plan_fft!, plan_bfft!
using LinearAlgebra: mul!
using FastBroadcast: @..
using Strided: Strided, @strided
using NestedThreading: NestedThreading

using MultiscaleFilterBanks: MultiscaleFilterBanks,
    data_eltype, filter_eltype, foreach_chunk, loop_threaded, with_fftw_threads, n_tasks,
    MaskBankEngine, MaskBank, MaskItem, mask_apply!, mask_accumulate!,
    mask_bank_forward!, mask_bank_inverse!

using MultiscaleTransformsBase: MultiscaleTransformsBase, AbstractMultiscaleTransform, AbstractTransformWorkspace,
    analyze, analyze!, synthesize, synthesize!, make_workspace, transform, similar_coefficients,
    subband_layout, nsubbands, subband, SubbandVector, SubbandLayout, check_workspace

# ── Types ─────────────────────────────────────────────────────────────────────
include("types/transforms.jl")
include("types/device.jl")

# ── Filter coefficients ───────────────────────────────────────────────────────
include("filters/bernstein.jl")
include("filters/checkerboard.jl")

# ── Pyramid stage ─────────────────────────────────────────────────────────────
include("pyramid/fourier_pyramid.jl")

# ── Directional filter bank ───────────────────────────────────────────────────
include("directional/hourglass.jl")
include("directional/ndfb_wedges.jl")
include("directional/ndfb.jl")
include("directional/ritf.jl")
include("directional/irc.jl")
include("directional/surfacelet_scale.jl")

# ── Top-level transforms ──────────────────────────────────────────────────────
include("transforms/ns_surfacelet.jl")
include("transforms/surfacelet.jl")

# ── Precompilation ────────────────────────────────────────────────────────────
include("precompile.jl")

# ── Exports ───────────────────────────────────────────────────────────────────
export
    # Transforms and workspaces
    NSSurfacelet, NSSurfaceletWorkspace,
    Surfacelet, SurfaceletWorkspace,
    surfacelet_redundancy, checkerboard_orders,

    # The interface (MultiscaleTransformsBase)
    AbstractMultiscaleTransform, AbstractTransformWorkspace,
    analyze, analyze!, synthesize, synthesize!, make_workspace, transform, similar_coefficients,
    subband_layout, nsubbands, subband, SubbandVector, SubbandLayout

# Not exported, but public: the directional stage on its own, and the pieces
# the hourglass bank is built from.
@compat public HourglassFB, make_hourglass, hourglass_mask, hourglass_weight,
    hourglass_decompose, hourglass_decompose!,
    hourglass_reconstruct, hourglass_reconstruct!,
    NDFB, make_ndfb, ndfb_sharing, ndfb_nsubbands, ndfb_subband_index,
    ndfb_subband_triple, ndfb_mask, ndfb_weight,
    ndfb_decompose, ndfb_decompose!,
    ndfb_reconstruct, ndfb_reconstruct!
@compat public bernstein_table, dft_frequencies, mapping_coefficients, mapping_response, halfband_split

end # module
