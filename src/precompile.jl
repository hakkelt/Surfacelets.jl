using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    # Small volume, low Bernstein order, and FFTW.ESTIMATE — a MEASURE plan would
    # time transforms during precompilation, and the wisdom it produced would be
    # for a size nobody asked for.
    vol = zeros(Float64, 8, 8, 8)
    vol[4, 4, 4] = 1.0
    params = NSurfaceletParams(J = 1, L_array = [1], order = 4)
    @compile_workload begin
        # The allocating round trip, the workspace path and the streaming form,
        # which is the entry point at working sizes.
        ws = make_nsurfacelet_workspace(params, size(vol), Float64; flags = FFTW.ESTIMATE)
        coeffs = nsurfacelet_forward(vol, params; workspace = ws)
        recoeffs = similar(vol)
        nsurfacelet_inverse!(recoeffs, coeffs, params; workspace = ws)
        nsurfacelet_forward(vol, params; workspace = ws) do j, k, band
            sum(band)
        end
    end
end
