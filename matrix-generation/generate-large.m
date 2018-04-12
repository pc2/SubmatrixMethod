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
