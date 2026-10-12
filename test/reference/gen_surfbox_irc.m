%% Generate golden reference data for the IRC-decimated surfacelet transform
%% from SurfBox (Yue Lu and Minh N. Do, MATLAB File Exchange 14485).
%%
%% Add SurfBox's Matlab/ directory to the path, then run this script from
%% test/reference/. On glnxa64 Surfdec takes its pure-MATLAB path, which is the
%% one this package mirrors stage by stage.
%%
%% To keep the repository small, inputs are not stored: they come from an exact
%% integer formula (`testvol` below, mirrored in test_irc.jl), and large outputs
%% are stored as fingerprints rather than in full. A fingerprint of an array A
%% is [Σ A.*W1, Σ A.*W2, Σ |A|.^2] with the fixed weights
%%     W1(i,j,k) = cos(0.37 i + 0.91 j + 1.53 k),
%%     W2(i,j,k) = sin(1.11 i - 0.29 j + 0.77 k)
%% (1-based indices); complex arrays store the two weighted sums as real and
%% imaginary parts, five numbers in all. Any coefficient error above ~1e-13 of
%% the array's norm changes a fingerprint.
%%
%% Every record goes into surfbox_irc.bin as little-endian Float64, and
%% surfbox_irc.txt gets one line per record:
%%
%%     name  kind  ndims  dim1 dim2 ...
%%
%% where kind is `full` (the array itself, column-major) or `fp` (the
%% fingerprint of an array of the listed size).
%%
%% SurfBox refuses a checkerboard stage whose block is narrower than its spatial
%% filter (29 x 29 at bo = 8, 13 x 13 at bo = 4) and an hourglass kernel (29
%% taps; Surfdec does not forward msize) wider than any scale, which is what
%% fixes the sizes and orders below.

fid = fopen('surfbox_irc.bin', 'w', 'ieee-le');
idx = fopen('surfbox_irc.txt', 'w');

%% Checkerboard filters, in full
for bo = [4 8 12]
    modes = {'d', 'dec'; 'r', 'rec'};
    for m = 1:2
        f = get_cbd_filters_load(bo, modes{m, 1});
        tags = {'H0', 'c0', 'H1', 'c1'};
        for t = 1:4
            putfull(fid, idx, sprintf('cbd_%d_%s_%s', bo, modes{m, 2}, tags{t}), f{t});
        end
    end
end

x32 = testvol(32);
x48 = testvol(48);
x64 = testvol(64);

%% Hourglass stage (ritf defaults: msize 15, beta 2.5, lambda 4)
hg = HourGlassDec(x32, 'S', 'F', 'ritf', 15, 2.5, 4);
for K = 1:3
    putfp(fid, idx, sprintf('hg_%d', K), hg{K});
end

%% IRC stage and the whole NDFB, bo = 4
levels = {[-1 1 1; 1 -1 1; 1 1 -1], [-1 2 1; 1 -1 2; 2 1 -1]};
names = {'l111', 'l212'};
Hd = get_cbd_filters_load(4, 'd');
for c = 1:2
    for K = 1:3
        putfp(fid, idx, sprintf('irc_%s_%d', names{c}, K), IRCdecF(hg{K}, K, levels{c}(K, :), Hd));
    end
    [Y, ri] = NDFBdec(x32, 'S', 'S', 'C', levels{c}, 'ritf', 'bo', 4);
    for K = 1:3
        for i = 1:numel(Y{K})
            putfp(fid, idx, sprintf('ndfb_%s_%d_%d', names{c}, K, i), Y{K}{i});
        end
    end
    putfp(fid, idx, sprintf('ndfbrec_%s', names{c}), NDFBrec(Y, 'S', ri));
end

%% Pyramid: mode 1.5 on 48^3, modes 1 and 2 on 32^3, two levels each
pyr = {48, 1.5, 'm15'; 32, 1, 'm1'; 32, 2, 'm2'};
for p = 1:3
    if pyr{p, 1} == 48, x = x48; else, x = x32; end
    s = PyrNDDec_mm(x, 'F', 2, pyr{p, 2}, @rcos);
    for j = 1:numel(s)
        putfp(fid, idx, sprintf('pyr_%s_%d', pyr{p, 3}, j), s{j});
    end
end

%% Whole transform: mode 1.5 on 48^3 (bo = 8) and mode 2 on 64^3 (bo = 4)
surf = {48, 1.5, {[-1 1 1; 1 -1 1; 1 1 -1], [-1 1 1; 1 -1 1; 1 1 -1]}, 'm15', 8;
        64, 2, {[-1 2 1; 1 -1 2; 2 1 -1], [-1 1 1; 1 -1 1; 1 1 -1]}, 'm2', 4};
for c = 1:2
    if surf{c, 1} == 48, x = x48; else, x = x64; end
    [Y, ri] = Surfdec(x, surf{c, 2}, surf{c, 3}, 'ritf', 'bo', surf{c, 5});
    for j = 1:numel(Y) - 1
        for K = 1:3
            for i = 1:numel(Y{j}{K})
                putfp(fid, idx, sprintf('surf_%s_%d_%d_%d', surf{c, 4}, j, K, i), Y{j}{K}{i});
            end
        end
    end
    putfp(fid, idx, sprintf('surf_%s_low', surf{c, 4}), Y{end});
    putfp(fid, idx, sprintf('surfrec_%s', surf{c, 4}), Surfrec(Y, ri));
end

%% Redundancy at 96^3, mode 1.5: 24 directional subbands per scale over three
%% scales, and 96 over two (a third 96-subband scale at 32^3 is too small for
%% even the bo = 4 filters).
x96 = testvol(96);
red = {[-1 1 2; 2 -1 1; 1 2 -1], 3, 'd24'; [-1 2 3; 3 -1 2; 2 3 -1], 2, 'd96'};
for c = 1:2
    lev = repmat({red{c, 1}}, 1, red{c, 2});
    Y = Surfdec(x96, 1.5, lev, 'ritf', 'bo', 4);
    n = numel(Y{end});
    for j = 1:numel(Y) - 1
        for K = 1:3
            for i = 1:numel(Y{j}{K})
                n = n + numel(Y{j}{K}{i});
            end
        end
    end
    putfull(fid, idx, sprintf('red_96_%s', red{c, 3}), n / numel(x96));
end

fclose(fid);
fclose(idx);
disp('done');

%% MATLAB requires local functions at the end of a script.

% Deterministic broadband test volume, exact in double precision.
function x = testvol(n)
    [i, j, k] = ndgrid(1:n, 1:n, 1:n);
    q = 73 * i.^2 + 179 * j.^2 + 283 * k.^2 + 31 * i .* j + 47 * j .* k + 61 * k .* i;
    x = mod(q, 1031) / 1031 - 0.5;
end

function putfull(fid, idx, name, A)
    A = double(A);
    fwrite(fid, A(:), 'double');
    fprintf(idx, '%s full %d %s\n', name, ndims(A), num2str(size(A)));
end

function putfp(fid, idx, name, A)
    A = double(A);
    [i, j, k] = ndgrid(1:size(A, 1), 1:size(A, 2), 1:size(A, 3));
    W1 = cos(0.37 * i + 0.91 * j + 1.53 * k);
    W2 = sin(1.11 * i - 0.29 * j + 0.77 * k);
    s1 = sum(A(:) .* W1(:));
    s2 = sum(A(:) .* W2(:));
    e = sum(abs(A(:)).^2);
    if isreal(A)
        v = [s1 s2 e];
    else
        v = [real(s1) imag(s1) real(s2) imag(s2) e];
    end
    fwrite(fid, v, 'double');
    fprintf(idx, '%s fp %d %s\n', name, ndims(A), num2str(size(A)));
end
