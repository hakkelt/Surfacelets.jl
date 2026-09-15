"""
    Surfacelets

Three-dimensional nonsubsampled and decimated surfacelet transforms.

## Quick reference

| Function | Description |
|----------|-------------|
| `nsurfacelet_forward` / `nsurfacelet_inverse` | Nonsubsampled surfacelet transform on volumes, with a streaming form |
| `surfacelet_forward` / `surfacelet_inverse` | Decimated counterpart, subbands at reduced resolution |
| `make_nsurfacelet_workspace` | Allocate a reusable workspace for volumes |
| `make_hourglass` / `make_ndfb` | The directional stage on its own |

The surfacelet transform is a construction in its own right, not a
dimension-raised contourlet transform.  Its directional stage is Lu & Do's
hourglass nonsubsampled directional filter bank: Bernstein mapping kernels carve
the spectrum into three hourglass-shaped regions around the three axes, each
refined into `2^L` wedges.  Lu & Do prove that an hourglass partition cannot be
critically sampled, which is why the construction has no separable-lattice
description — there is no quincunx sampling matrix, no shear index remap and no
ladder filter anywhere in it.  See the "Theory" page.

`surfacelet_forward` is the decimated counterpart of `nsurfacelet_forward`: the
same frequency partition, but each subband is stored at its own reduced
resolution (`make_surfacelet`, `surfacelet_redundancy`).  It trades shift
invariance for a roughly 3× smaller coefficient set.
"""
module Surfacelets

using FFTW: FFTW, plan_rfft, plan_fft, plan_fft!, plan_irfft, plan_ifft, plan_ifft!, rfft, fft
using LinearAlgebra: mul!, adjoint

using MultiscaleFilterBanks: MultiscaleFilterBanks,
    FilterPair, CDF97, upsample_filter,
    ThreadingPolicy, Auto, Enabled, Disabled,
    _data_eltype, _filter_eltype, _use_threading, _clamp_idx,
    _nsp_atrous_filters, _NSP_SYNTH_SCALE

# ── Types ─────────────────────────────────────────────────────────────────────
include("types.jl")

# ── Filter coefficients ───────────────────────────────────────────────────────
include("filters/bernstein.jl")

# ── DSP primitives ────────────────────────────────────────────────────────────
include("primitives/conv3d_fft.jl")
include("primitives/conv3d.jl")

# ── Pyramid stage ─────────────────────────────────────────────────────────────
include("pyramid/nonsubsampled_pyramid3d.jl")

# ── Directional filter bank ───────────────────────────────────────────────────
include("directional/hourglass.jl")
include("directional/ndfb.jl")

# ── Top-level transforms ──────────────────────────────────────────────────────
include("transforms/nsurfacelet.jl")
include("transforms/surfacelet.jl")

# ── Precompilation ────────────────────────────────────────────────────────────
include("precompile.jl")

# ── Exports ───────────────────────────────────────────────────────────────────
export
    # Re-exported from MultiscaleFilterBanks
    FilterPair, CDF97, upsample_filter,
    ThreadingPolicy, Auto, Enabled, Disabled,

    # Types
    NSurfaceletParams, NSurfaceletCoefficients, NSurfaceletWorkspace,
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
    nsurfacelet_forward, nsurfacelet_inverse, nsurfacelet_inverse!, nsurfacelet_nsubbands,
    make_nsurfacelet_workspace,
    surfacelet_forward, surfacelet_inverse, surfacelet_nsubbands,
    surfacelet_redundancy, make_surfacelet

end # module
