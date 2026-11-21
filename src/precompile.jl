using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    # Small volume, low Bernstein order, and FFTW.ESTIMATE — a MEASURE plan would
    # time transforms during precompilation, and the wisdom it produced would be
    # for a size nobody asked for.
    vol = zeros(Float64, 8, 8, 8)
    vol[4, 4, 4] = 1.0
    t = NSSurfacelet(direction_levels = [1], bernstein_order = 4)
    @compile_workload begin
        # The allocating round trip, the workspace path and the streaming form,
        # which is the entry point at working sizes.
        ws = make_workspace(t, Float64, size(vol); flags = FFTW.ESTIMATE)
        coeffs = analyze(t, vol; workspace = ws)
        synthesize!(similar(vol), t, coeffs; workspace = ws)
        analyze(t, vol; workspace = ws) do i, band
            sum(band)
        end
    end
end
