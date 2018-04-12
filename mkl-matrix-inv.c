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
#include <time.h>
#include "timespec_subtract.h"
#include "matrix_io.h"

#define CLOCK_ID CLOCK_MONOTONIC

int invert_matrix(double *matrix, long size) {
  // First we need to compute the LU factorization using ?getrf
  int *ipiv, ret;
  ipiv = (int*) mkl_calloc(size, sizeof(int), 64);

  ret = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, size, size, matrix, size, ipiv);
  if (ret) {
    mkl_free(ipiv);
    return ret;
  }

  // And now we calculate the inverse using the LU factorization
  ret = LAPACKE_dgetri(LAPACK_ROW_MAJOR, size, matrix, size, ipiv);
  mkl_free(ipiv);
  return ret;
}

int main (int argc, char* argv[]) {
  long size, *nnz;
  char *fn_in, *fn_out;
  double *matrix;
  struct timespec before, after, diff;

  if (argc != 4) {
    fprintf (stderr,
      "Usage: ./mkl-matrix-inv matrix_size input-file.txt output-file.txt\n");
    exit(EXIT_FAILURE);
  }

  size = strtol(argv[1], NULL, 10);
  fn_in  = argv[2];
  fn_out = argv[3];

  matrix = (double*) mkl_calloc(size*size, sizeof(double), 64);
  nnz = (long*) calloc(size, sizeof(long));

  clock_gettime(CLOCK_ID, &before);
  read_input_matrix_d(matrix, nnz, size, fn_in);
  clock_gettime(CLOCK_ID, &after);
  timespec_subtract(&diff, &after, &before);
  printf("Reading input matrix took %.3fs\n",
    1.0*diff.tv_sec + 1e-9*diff.tv_nsec);

  clock_gettime(CLOCK_ID, &before);
  invert_matrix(matrix, size);
  clock_gettime(CLOCK_ID, &after);
  timespec_subtract(&diff, &after, &before);
  printf("Doing a precise inversion took %.3fs\n",
    1.0*diff.tv_sec + 1e-9*diff.tv_nsec);

  clock_gettime(CLOCK_ID, &before);
  write_output_matrix_d(matrix, size, fn_out);
  clock_gettime(CLOCK_ID, &after);
  timespec_subtract(&diff, &after, &before);
  printf("Writing result matrix took %.3fs\n",
    1.0*diff.tv_sec + 1e-9*diff.tv_nsec);

  free(nnz);
  mkl_free(matrix);
  exit(EXIT_SUCCESS);
}
