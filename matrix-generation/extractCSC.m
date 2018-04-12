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
