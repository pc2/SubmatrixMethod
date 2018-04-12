/*
 * MIT License
 * 
 * Copyright (c) 2018 Paderborn Center for Parallel Computing
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <fcntl.h>
#include <mkl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#include "matrix_io.h"

#define PATHLEN 255

int main(int argc, char* argv[]) {

  long size, total_nnz;
  MKL_INT *row_ind, *col_ptr; //CSC
  MKL_INT *col_ind, *row_ptr; //CSR
  MKL_INT ret, intsize;
  double *matrix, *csrval, *cscval;
  char fn_in_val[PATHLEN], fn_in_ri[PATHLEN], fn_in_cp[PATHLEN];
  FILE *fp;
  int fd;

  if (argc != 4) {
    fprintf(stderr,
      "Usage: ./csc-to-matlab matrix_size input-name output-file.txt\n");
    exit(EXIT_FAILURE);
  }
  size = strtol(argv[1], NULL, 10);
  intsize = (MKL_INT)size;
  snprintf(fn_in_val, PATHLEN, "%s.val", argv[2]);
  snprintf(fn_in_ri, PATHLEN, "%s.ri", argv[2]);
  snprintf(fn_in_cp, PATHLEN, "%s.cp", argv[2]);

  fd = open(fn_in_cp, O_RDONLY);
  col_ptr = (MKL_INT*) mmap(NULL, (size+1)*sizeof(MKL_INT), PROT_READ,
    MAP_SHARED, fd, 0);
  close(fd);
  total_nnz = col_ptr[size];
  fd = open(fn_in_ri, O_RDONLY);
  row_ind = (MKL_INT*) mmap(NULL, total_nnz*sizeof(MKL_INT), PROT_READ,
    MAP_SHARED, fd, 0);
  close(fd);
  fd = open(fn_in_val, O_RDONLY);
  cscval = (double*) mmap(NULL, total_nnz*sizeof(double), PROT_READ,
    MAP_SHARED, fd, 0);
  close(fd);

  matrix = (double*) mkl_calloc(size*size, sizeof(double), 64);
  csrval = (double*) mkl_calloc(total_nnz, sizeof(double), 64);
  col_ind = (MKL_INT*) mkl_calloc(total_nnz, sizeof(MKL_INT), 64);
  row_ptr = (MKL_INT*) mkl_calloc(size+1, sizeof(MKL_INT), 64);

  /* Now we do the inverse of what we do inn matlab-to-csc -- first to CSR and
   * then to full matrix format. */
  MKL_INT job1[6] = {1,0,0,0,0,1};
  mkl_dcsrcsc (job1, &intsize, csrval, col_ind, row_ptr, cscval, row_ind,
    col_ptr, &ret);
  MKL_INT job2[6] = {1,0,0,2,0,1};
  mkl_ddnscsr (job2, &intsize, &intsize, matrix, &intsize, csrval, col_ind,
    row_ptr, &ret);
  if (ret != 0) {
    fprintf(stderr, "Oh oh, something went wrong :(\n");
  }

  write_output_matrix_d(matrix, size, argv[3]);

  mkl_free(row_ptr);
  mkl_free(col_ind);
  mkl_free(csrval);
  mkl_free(matrix);
  munmap(cscval, total_nnz*sizeof(double));
  munmap(row_ind, total_nnz*sizeof(MKL_INT));
  munmap(col_ptr, (size+1)*sizeof(MKL_INT));
  exit(EXIT_SUCCESS);
}
