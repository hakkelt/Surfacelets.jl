# [API Reference](@id api)

```@meta
CurrentModule = Surfacelets
```

## Module

```@docs
Surfacelets
MultiscaleFilterBanks.MultiscaleFilterBanks
```

## Nonsubsampled surfacelet transform

```@docs
ns_surfacelet_forward
ns_surfacelet_forward!
ns_surfacelet_inverse
ns_surfacelet_inverse!
ns_surfacelet_nsubbands
make_ns_surfacelet_workspace
NSSurfaceletParams
NSSurfaceletCoefficients
NSSurfaceletWorkspace
```

## Decimated surfacelet transform

```@docs
surfacelet_forward
surfacelet_forward!
surfacelet_inverse
surfacelet_inverse!
surfacelet_nsubbands
surfacelet_redundancy
make_surfacelet
SurfaceletParams
SurfaceletCoefficients
SurfaceletPlan
checkerboard_orders
```

## Semi-translation-invariant (STI) surfacelet transform

```@docs
sti_surfacelet_forward
sti_surfacelet_forward!
sti_surfacelet_inverse
sti_surfacelet_inverse!
sti_surfacelet_nsubbands
make_sti_surfacelet
STISurfaceletParams
```

## Directional filter bank

```@docs
HourglassFB
make_hourglass
hourglass_mask
hourglass_weight
hourglass_decompose
hourglass_decompose!
hourglass_reconstruct
hourglass_reconstruct!
NDFB
make_ndfb
ndfb_sharing
ndfb_nsubbands
ndfb_subband_index
ndfb_subband_triple
ndfb_mask
ndfb_weight
ndfb_decompose
ndfb_decompose!
ndfb_reconstruct
ndfb_reconstruct!
```

## Pyramid

```@docs
nsp3d_decompose
nsp3d_decompose!
nsp3d_reconstruct
nsp3d_reconstruct!
```

## Primitives

```@docs
conv3d_sep
conv3d_sep!
conv3d_sep_fft
conv3d_sep_fft!
Conv3DFFTPlan
make_conv3d_fft_plan
```

## Filters

Re-exported from
[MultiscaleFilterBanks.jl](https://github.com/hakkelt/MultiscaleFilterBanks.jl),
the shared core, so `using Surfacelets` alone is enough.

```@docs
CDF97
FilterPair
upsample_filter
```

## Threading

```@docs
ThreadingPolicy
Auto
Enabled
Disabled
```

## Filter design internals

Not exported — these are the pieces the hourglass bank is built from, useful if
you want to design your own mapping kernels.

```@docs
bernstein_table
dft_frequencies
mapping_coefficients
mapping_response
halfband_split
```
