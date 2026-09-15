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
#     ws  = Adapt.adapt(CuArray, make_nsurfacelet_workspace(params, dims, Float32))
#
# FFTW plans cannot be moved, so each is replanned against the adapted buffer;
# `plan_rfft` is the AbstractFFTs generic, so it dispatches to the backend's FFT
# implementation.  Planning happens once per adapt, not per call.
#
# The 1-D à trous kernel transforms in `Conv3DFFTPlan` stay on the host: they are
# length-`n` and the result is three vectors, so a device FFT would cost more in
# launch overhead than it saves.  Only the transfer-function arrays live on the
# device.

using Surfacelets: HourglassFB, NDFB, NSurfaceletWorkspace, NSurfaceletCoefficients, Conv3DFFTPlan

function Adapt.adapt_structure(to, fb::HourglassFB)
    m1 = Adapt.adapt(to, fb.m1)
    m2 = Adapt.adapt(to, fb.m2)
    rbuf = Adapt.adapt(to, fb.rbuf)
    fwd = Surfacelets.plan_rfft(rbuf)
    spec = fwd * rbuf
    inv = Surfacelets.plan_irfft(spec, size(rbuf, 1))
    T = eltype(m1)
    return HourglassFB{T, typeof(m1), typeof(spec), typeof(fwd), typeof(inv)}(
        fb.dims, fb.order, fb.poly, fb.tight, m1, m2, fwd, inv, spec, similar(spec), rbuf,
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

function Adapt.adapt_structure(to, p::Conv3DFFTPlan)
    ker = Adapt.adapt(to, p.ker)
    fwd = Surfacelets.plan_rfft(ker)
    freq = fwd * ker
    inv = Surfacelets.plan_irfft(freq, size(ker, 1))
    T = eltype(ker)
    return Conv3DFFTPlan{T, typeof(ker), typeof(freq), typeof(fwd), typeof(inv)}(
        p.dims, fwd, inv, freq, ker,
        Adapt.adapt(to, p.w1), Adapt.adapt(to, p.w2), Adapt.adapt(to, p.w3),
        p.kbuf1, p.kbuf2, p.kbuf3,
        p.aux === nothing ? nothing : similar(ker),
    )
end

function Adapt.adapt_structure(to, ws::NSurfaceletWorkspace)
    hg = Adapt.adapt(to, ws.banks[1].fb)
    banks = [_adapt_ndfb(to, b, hg) for b in ws.banks]
    conv = Adapt.adapt(to, ws.conv)
    input = Adapt.adapt(to, ws.input)
    Td = eltype(input)
    T = eltype(hg.m1)
    return NSurfaceletWorkspace{Td, T, eltype(banks), typeof(conv), typeof(input)}(
        ws.dims, banks, conv, ws.lp_filters, ws.filters, input,
        Adapt.adapt(to, ws.coarse), Adapt.adapt(to, ws.bandpass),
        Adapt.adapt(to, ws.tmp), Adapt.adapt(to, ws.tmp2),
    )
end

function Adapt.adapt_structure(to, c::NSurfaceletCoefficients)
    return NSurfaceletCoefficients(
        Adapt.adapt(to, c.coarse),
        [[Adapt.adapt(to, s) for s in lvl] for lvl in c.subbands],
    )
end
