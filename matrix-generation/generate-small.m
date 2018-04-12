% This script expects that size, density, cond, rep and kind is set by
% the calling script. The result matrix is stored in text format.

filename = strcat('sprandsym-s', num2str(size), '-d', num2str(density), '-c', num2str(cond), '-n', num2str(rep), '.txt');
rng('shuffle')
M = sprandsym(size, density/100, 1/cond, kind);
dlmwrite(filename, full(M));
