"""
    Surfacelets

Three-dimensional nonsubsampled, decimated and semi-translation-invariant
surfacelet transforms.

## Quick reference

| Function | Description |
|----------|-------------|
| `ns_surfacelet_forward` / `ns_surfacelet_inverse` | Nonsubsampled surfacelet transform on volumes, with a streaming form |
| `surfacelet_forward` / `surfacelet_inverse` | Decimated counterpart, subbands at reduced resolution |
| `sti_surfacelet_forward` / `sti_surfacelet_inverse` | Semi-translation-invariant (STI): nonsubsampled pyramid, decimated directional stage |
| `make_ns_surfacelet_workspace` | Allocate a reusable workspace for volumes |
| `make_hourglass` / `make_ndfb` | The directional stage on its own |

The surfacelet transform is a construction in its own right, not a
dimension-raised contourlet transform.  Its first directional stage carves the
spectrum into three hourglass-shaped regions around the three axes, a partition
with no quincunx-lattice description that Lu & Do prove cannot be critically
sampled.  The nonsubsampled transform builds it from Bernstein mapping kernels
and cuts each hourglass into `2^L` wedges with directly placed boundaries.

`surfacelet_forward` is Lu & Do's decimated surfacelet transform, reproducing
their reference toolbox SurfBox to rounding error: a 3/2-resampling Fourier
pyramid, SurfBox's hourglass bank, and iterated resampled checkerboard filter
banks that refine each hourglass into wedges while decimating critically
(`make_surfacelet`, `surfacelet_redundancy`).  About 4 coefficients per voxel,
whatever the number of directions; not shift-invariant.  See the "Theory" page.
"""
module Surfacelets

using Compat: @compat
using FFTW: FFTW, plan_rfft, plan_irfft, plan_brfft, plan_fft!, plan_bfft!, rfft, fft
using LinearAlgebra: mul!, adjoint
using FastBroadcast: @..
using Strided: Strided, @strided

using MultiscaleFilterBanks: MultiscaleFilterBanks,
    FilterPair, CDF97, upsample_filter,
    data_eltype, filter_eltype, use_threading, clamp_idx,
    nsp_atrous_filters, nsp_responses, foreach_chunk, add_partials!,
    loop_threaded, with_fftw_threads

# ── Types ─────────────────────────────────────────────────────────────────────
include("types/params.jl")
include("types/coefficients.jl")

# ── Filter coefficients ───────────────────────────────────────────────────────
include("filters/bernstein.jl")
include("filters/checkerboard.jl")

# ── DSP primitives ────────────────────────────────────────────────────────────
include("primitives/conv3d_fft.jl")
include("primitives/conv3d.jl")

# ── Pyramid stage ─────────────────────────────────────────────────────────────
include("pyramid/nonsubsampled_pyramid3d.jl")
include("pyramid/fourier_pyramid.jl")
include("pyramid/raised_cosine_pyramid.jl")

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
include("transforms/sti_surfacelet.jl")

# ── Precompilation ────────────────────────────────────────────────────────────
include("precompile.jl")

# ── Exports ───────────────────────────────────────────────────────────────────
export
    # Re-exported from MultiscaleFilterBanks
    FilterPair, CDF97, upsample_filter,

    # Types
    NSSurfaceletParams, NSSurfaceletCoefficients, NSSurfaceletWorkspace,
    SurfaceletParams, SurfaceletCoefficients, SurfaceletPlan,

    # Primitives
    conv3d_sep!, conv3d_sep, conv3d_sep_fft!, conv3d_sep_fft,
    Conv3DFFTPlan, make_conv3d_fft_plan,

    # Building blocks
    nsp3d_decompose, nsp3d_decompose!, nsp3d_reconstruct, nsp3d_reconstruct!,
    HourglassFB, make_hourglass, hourglass_mask, hourglass_weight,
    hourglass_decompose, hourglass_decompose!,
    hourglass_reconstruct, hourglass_reconstruct!,
    NDFB, make_ndfb, ndfb_sharing, ndfb_nsubbands, ndfb_subband_index,
    ndfb_subband_triple, ndfb_mask, ndfb_weight,
    ndfb_decompose, ndfb_decompose!,
    ndfb_reconstruct, ndfb_reconstruct!,

    # Top-level transforms
    ns_surfacelet_forward, ns_surfacelet_forward!, ns_surfacelet_inverse, ns_surfacelet_inverse!,
    ns_surfacelet_nsubbands,
    make_ns_surfacelet_workspace,
    surfacelet_forward, surfacelet_forward!, surfacelet_inverse, surfacelet_inverse!,
    surfacelet_nsubbands, surfacelet_redundancy, make_surfacelet, checkerboard_orders,
    STISurfaceletParams,
    sti_surfacelet_forward, sti_surfacelet_forward!, sti_surfacelet_inverse, sti_surfacelet_inverse!,
    sti_surfacelet_nsubbands, make_sti_surfacelet

# Not exported, but public: the pieces the hourglass bank is built from.
@compat public bernstein_table, dft_frequencies, mapping_coefficients, mapping_response, halfband_split

end # module
