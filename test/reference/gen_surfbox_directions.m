%% Generate golden reference data for the 3-D directional filter bank from SurfBox.
%%
%% Requires SurfBox (MATLAB File Exchange submission 14485) on the path; run
%% from test/reference/.
%%
%% For each of 80 pseudo-random directions this drives a plane wave through
%% SurfBox's NDFBdec at 32^3 with Level = [-1 1 1; 1 -1 1; 1 1 -1] (12
%% directional subbands, matching this package's `levels = 2`) and records
%% which subband carries the most energy.
%%
%% Writes surfbox_directions_32.bin: 80 direction vectors (3 x 80) followed by
%% 80 subband indices, all little-endian Float64 in column-major order.

rng(7);
N3 = 32; K = 1;
Level = [-1 K K; K -1 K; K K -1];
nd = 80;

dirs = randn(3, nd);
dirs = dirs ./ vecnorm(dirs);
assign = zeros(nd, 1);

[x1, x2, x3] = ndgrid(0:N3-1, 0:N3-1, 0:N3-1);
for t = 1 : nd
    kk = round(dirs(:, t) * 0.6 * N3 / 2);
    if all(kk == 0), kk = [1; 0; 0]; end
    Xv = cos(2 * pi * (kk(1) * x1 + kk(2) * x2 + kk(3) * x3) / N3);
    Y = NDFBdec(Xv, 'S', 'S', 'C', Level, 'ritf', 'bo', 8);
    best = -1; idx = 0;
    for h = 1 : 3
        for d = 1 : numel(Y{h})
            idx = idx + 1;
            e = sum(abs(Y{h}{d}(:)) .^ 2);
            if e > best, best = e; assign(t) = idx; end
        end
    end
end

fid = fopen('surfbox_directions_32.bin', 'w');
fwrite(fid, dirs, 'double');
fwrite(fid, assign, 'double');
fclose(fid);
fprintf('wrote surfbox_directions_32.bin (%d subbands)\n', idx);
