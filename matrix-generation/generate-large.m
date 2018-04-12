% MIT License
% 
% Copyright (c) 2018 Paderborn Center for Parallel Computing
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

rng('shuffle')
cond = 2;

for rep = 1:3;
  for exp = 11:1:18;
          size = 2^exp;
          density = 1000000;
          prefix = strcat('sprandsym-s', num2str(size), '-d', num2str(density), '-c', num2str(cond), '-n', num2str(rep));
          A = sprandsym(size, density/100000000, 1/cond, 2);
          [val,row_ind,col_ptr] = extractCSC(A, prefix);
  end;

  for exp = 10:1:20;
    size = 2^exp;
    density = 16000000 * 1024 / size;
    prefix = strcat('sprandsym-s', num2str(size), '-d', num2str(density), '-c', num2str(cond), '-n', num2str(rep));
    A = sprandsym(size, density/100000000, 1/cond, 2);
    [val,row_ind,col_ptr] = extractCSC(A, prefix);
  end;
end;
