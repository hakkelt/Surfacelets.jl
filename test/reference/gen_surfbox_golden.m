%% Generate golden reference data for the 3-D hourglass filter bank from SurfBox.
%%
%% SurfBox (Yue Lu and Minh N. Do, UIUC) is available from MATLAB File Exchange
%% submission 14485 ("Surfacelet Toolbox"). Add its Matlab/ directory to the
%% path, then run this script from test/reference/.
%%
%% It writes surfbox_hourglass_32.bin: the first hourglass analysis mask V_1 of
%% HourGlassDec on a 32^3 grid, restricted to the non-negative-frequency octant
%% (17 x 17 x 17) and stored as little-endian Float32 in column-major order.
%% The other two masks are axis permutations of this one, and the mask is even
%% along every axis, so the octant determines the full 32^3 array.
%%
%% Parameters are HourGlassDec's 'ritf' defaults: msize = 15, beta = 2.5,
%% lambda = 4. The masks satisfy sum_k V_k^2 = 3 exactly (a tight frame with
%% redundancy 3).

N3 = 32; szX = [N3 N3 N3]; N = 3;
msize = 15; beta = 2.5; lambda = 4;

mdiamond = NXdiamondmapping(msize, beta);
Ldiamond = 2 * msize - 1;

diamond_array = cell(N - 1, N - 1);
for k = 1 : N - 1
    for m = k + 1 : N
        f = zeros(szX([k m]));
        f(1:Ldiamond, 1:Ldiamond) = mdiamond;
        diamond_array{k, m - 1} = ...
            realpow(real(fft2(circshift(f, 1 - [msize msize]'))), lambda);
    end
end

hgc = cell(N, 1);
denom = zeros(szX);
for k = 1 : N
    cumstrt = 0;
    for m = 1 : N
        if m == k, continue; end
        sz = ones(1, N); sz([k m]) = szX([k m]);
        p = diamond_array{min(k, m), max(k, m) - 1};
        if k < m
            p = circshift(p, [size(p, 1) / 2, 0]);
        else
            p = circshift(p, [0, size(p, 2) / 2]);
        end
        p = repmat(reshape(p, sz), szX ./ sz);
        if ~cumstrt, hgc{k} = p; cumstrt = 1; else hgc{k} = hgc{k} .* p; end
    end
    denom = denom + hgc{k};
end
denom = N ./ denom;

V1 = realsqrt(hgc{1} .* denom);

S = zeros(szX);
for k = 1 : N
    S = S + hgc{k} .* denom;
end
fprintf('sum V^2 in [%.12f, %.12f]\n', min(S(:)), max(S(:)));

half = N3 / 2 + 1;
octant = single(V1(1:half, 1:half, 1:half));
fid = fopen('surfbox_hourglass_32.bin', 'w');
fwrite(fid, octant, 'single');
fclose(fid);
fprintf('wrote surfbox_hourglass_32.bin (%d values)\n', numel(octant));
