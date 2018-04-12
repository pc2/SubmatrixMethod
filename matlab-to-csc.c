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

#include <mkl.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrix_io.h"

#define PATHLEN 255

int main(int argc, char* argv[]) {
  
  long size, total_nnz, i, *nnz;
  MKL_INT *row_ind, *col_ptr; //CSC
  MKL_INT *col_ind, *row_ptr; //CSR
  MKL_INT ret, intsize;
  double *matrix, *csrval, *cscval;
  char fn_out_val[PATHLEN], fn_out_ri[PATHLEN], fn_out_cp[PATHLEN];
  FILE *fp;
  
  if (argc != 4) {
    fprintf(stderr,
      "Usage: ./matlab-to-csc matrix_size input-file.txt output-name\n");
    exit(EXIT_FAILURE);
  }
  size = strtol(argv[1], NULL, 10);
  intsize = (MKL_INT)size;
  
  matrix = (double*) mkl_calloc(size*size, sizeof(double), 64);
  nnz = (long*) calloc(size, sizeof(long));
  col_ptr = (MKL_INT*) mkl_calloc(size+1, sizeof(MKL_INT), 64);
  row_ptr = (MKL_INT*) mkl_calloc(size+1, sizeof(MKL_INT), 64);
  read_input_matrix_d(matrix, nnz, size, argv[2]);
  
  total_nnz = 0;
  for (i = 0; i < size; i++) {
    total_nnz += nnz[i];
  }
  csrval = (double*) mkl_calloc(total_nnz, sizeof(double), 64);
  cscval = (double*) mkl_calloc(total_nnz, sizeof(double), 64);
  row_ind = (MKL_INT*) mkl_calloc(total_nnz, sizeof(MKL_INT), 64);
  col_ind = (MKL_INT*) mkl_calloc(total_nnz, sizeof(MKL_INT), 64);
  
  /* We will use mkl_?dnscsr to convert our matrix into CSR and afterwards
   * mkl_?csrcsc to transform it into CSC. The second step should be a noop
   * in our case as the matrices are symmetric. */
  MKL_INT job1[6] = {0,0,0,2,total_nnz,1};
  mkl_ddnscsr(job1, &intsize, &intsize, matrix, &intsize, csrval, col_ind, row_ptr,
    &ret);
  if (ret != 0) {
    fprintf(stderr, "Oh oh, something went wrong :(\n");
    exit(EXIT_FAILURE);
  }
  MKL_INT job2[6] = {0,0,0,0,0,1};
  mkl_dcsrcsc(job2, &intsize, csrval, col_ind, row_ptr, cscval, row_ind,
    col_ptr, &ret);

  /* Write binary data into three separate files. So we don't need to think
   * about some binary file format. */
  snprintf(fn_out_val, PATHLEN, "%s.val", argv[3]);
  snprintf(fn_out_ri, PATHLEN, "%s.ri", argv[3]);
  snprintf(fn_out_cp, PATHLEN, "%s.cp", argv[3]);
  fp = fopen(fn_out_val, "wb");
  fwrite(cscval, sizeof(double), total_nnz, fp);
  fclose(fp);
  fp = fopen(fn_out_ri, "wb");
  fwrite(row_ind, sizeof(MKL_INT), total_nnz, fp);
  fclose(fp);
  fp = fopen(fn_out_cp, "wb");
  fwrite(col_ptr, sizeof(MKL_INT), size+1, fp);
  fclose(fp);
  
  mkl_free(col_ind);
  mkl_free(row_ind);
  mkl_free(cscval);
  mkl_free(csrval);
  mkl_free(row_ptr);
  mkl_free(col_ptr);
  free(nnz);
  mkl_free(matrix);
  exit(EXIT_SUCCESS);
}
