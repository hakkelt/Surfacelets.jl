# Adapt extension notes

- **GPU needs no kernels.** Every transform is FFT plus pointwise masks, and
  every mask is stored as *broadcastable factors* — the mapping responses and wedge
  planes are 3-D arrays with singleton axes where they do not depend on a frequency
  — so mask application, the complex real/imaginary split, and the pyramid's
  frequency multiply are all plain broadcasts. Keep it that way: introducing scalar
  indexing into `_mask_multiply!`, `_mask_accumulate!` or `_apply_freq_filter!`
  silently breaks the device path. `ext/` therefore contains only
  `Adapt.adapt_structure` methods, which move the data and replan the FFTs (plans
  cannot be adapted), which is why `Adapt` is the only weak dependency — no
  GPUArrays, no KernelAbstractions. A workspace adapt moves the shared hourglass
  response **once**, not once per scale. JLArrays is excluded from the GPU tests: it
  has no FFT implementation, so `plan_rfft` falls back to a scalar-indexing
  generic. CUDA and OpenCL are verified.
  `surfacelet` and `sti_surfacelet` follow the same rule: the pyramid is a
  broadcast against its separable profile plus a crop through index views, the
  hourglass stage a broadcast against precomputed masks, and each IRC stage a
  broadcast between the two halves of a reshaped view. Their plan is built on the
  host — the checkerboard filter slices are evaluated point by point in closed
  form — and then adapted, which moves the tables and rebuilds the buffers and FFT
  plans on the device; a device volume with no plan is an `ArgumentError`, not
  an implicit rebuild. Adapt the plan to the *bare* array type
  (`Adapt.adapt(CuArray, pl)`), never `typeof(x)`: it holds real, complex and
  integer tables, and an element-typed destination cannot fit them all. Both are
  verified on CUDA.
- **Deliberately not CUDA-graph-captured.** The forward/inverse are streaming —
  they hand each subband to a caller's callback, which is arbitrary host code, and
  stream capture admits none — and there is nothing to win anyway: this is a few
  full-volume FFTs, where launch overhead is noise.
