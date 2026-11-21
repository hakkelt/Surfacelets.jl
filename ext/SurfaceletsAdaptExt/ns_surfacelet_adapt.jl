# 3-D stack on the device.
#
# Nothing here is a kernel.  The 3-D transforms are FFT plus a pointwise mask,
# and every mask in the 3-D stack is stored as broadcastable factors — the
# mapping responses and wedge planes are 3-D arrays with singleton axes where
# they do not depend on a frequency — so the mask application, the complex
# real/imaginary split and the pyramid's frequency-domain multiply are all plain
# broadcasts already.  What the device needs is therefore only the data moved
# across and the FFT plans rebuilt for device arrays, which is what these
# `Adapt.adapt_structure` methods do:
#
#     fb  = Adapt.adapt(CuArray, make_hourglass(dims, Float32))
#     ws  = Adapt.adapt(CuArray, make_workspace(NSSurfacelet(...), Float32, dims))
#
# FFTW plans cannot be moved, so each is replanned against the adapted buffer;
# `plan_rfft` is the AbstractFFTs generic, so it dispatches to the backend's FFT
# implementation.  Planning happens once per adapt, not per call.

using Surfacelets: HourglassFB, NDFB, NSSurfaceletWorkspace

function Adapt.adapt_structure(to, fb::HourglassFB)
    m1 = Adapt.adapt(to, fb.m1)
    m2 = Adapt.adapt(to, fb.m2)
    rbuf = Adapt.adapt(to, fb.rbuf)
    fwd = plan_rfft(rbuf)
    spec = fwd * rbuf
    inv = plan_irfft(spec, size(rbuf, 1))
    T = eltype(m1)
    return HourglassFB{T, typeof(m1), typeof(spec), typeof(fwd), typeof(inv)}(
        fb.dims, fb.bernstein_order, fb.halfband_order, fb.tight, m1, m2, fwd, inv, spec, similar(spec), rbuf,
        fb.spec2 === nothing ? nothing : similar(spec),
        fb.rbuf2 === nothing ? nothing : similar(rbuf),
        fb.cbuf === nothing ? nothing : similar(spec, fb.dims),
    )
end

# Rebuild a directional bank around an already-adapted hourglass bank, so a
# multiscale workspace does not end up with one copy of the (large) mapping
# response per scale.
function _adapt_ndfb(to, fb::NDFB, hg::HourglassFB)
    wedges = map(fb.wedges) do planes
        map(planes) do masks
            [Adapt.adapt(to, w) for w in masks]
        end
    end
    A3 = typeof(wedges[1][1][1])
    return NDFB{eltype(hg.m1), A3, typeof(hg)}(hg, fb.levels, wedges)
end

Adapt.adapt_structure(to, fb::NDFB) = _adapt_ndfb(to, fb, Adapt.adapt(to, fb.fb))

# The workspace's tables move; its buffers and FFT plans are rebuilt next to them.
# Adapt to the bare array type (`CuArray`), not an element-typed one, so the
# tables keep the workspace's precision.
function Adapt.adapt_structure(to, ws::NSSurfaceletWorkspace{Td, T}) where {Td, T}
    ad(x) = Adapt.adapt(to, x)
    wedges = [map(hg -> map(planes -> [ad(w) for w in planes], hg), s) for s in ws.wedges]
    resp(v) = [map(ad, axes3) for axes3 in v]
    proto = ad(Array{T, 3}(undef, 0, 0, 0))
    return Surfacelets._ns_workspace(
        Td, ws.dims, ws.transform, ad(ws.m1), ad(ws.m2), wedges, resp(ws.pa), resp(ws.pl),
        typeof(ad(ws.m1))[ad(b) for b in ws.band], proto, FFTW.ESTIMATE, ws.tasks
    )
end
