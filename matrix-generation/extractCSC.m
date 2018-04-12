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

function [val,row_ind,col_ptr] = extractCSC(A, prefix)

	n = size(A,1);
	nonzero = nnz(A);
	[row_ind,j,val] = find(A);

	col_ptr = zeros(n+1, 1);
	col_ptr(1) = 0;
	for m = 2:n
		x = find(j==m, 1);
		if (isempty(x))
			col_ptr(m) = -1;
		else
			col_ptr(m) = x-1;
		end
	end
	col_ptr(n+1) = nonzero;

	for m = n:-1:1
		if (col_ptr(m) == -1)
			col_ptr(m) = col_ptr(m+1);
		end
	end

	for m = 1:nonzero
		row_ind(m) = row_ind(m) - 1;
	end

	filename = strcat(prefix, '.val');
	fid = fopen(filename, 'w');
	fwrite(fid, val, 'double');
	fclose(fid);

	filename = strcat(prefix, '.ri');
	fid = fopen(filename, 'w');
	fwrite(fid, row_ind, 'int32');
	fclose(fid);

	filename = strcat(prefix, '.cp');
	fid = fopen(filename, 'w');
	fwrite(fid, col_ptr, 'int32');
	fclose(fid);

end
