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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "matrix_io.h"

template <typename T> void read_input_matrix_templ(T *matrix_in, long *nnz,
  long size, char *fn_in) {

  long i, j;
  char buf[BUF_SIZE];
  char *bufp, *token;
  FILE *stream_in;
  T val;

  stream_in = fopen(fn_in, "r");
  i = 0;
  while (fgets(buf, BUF_SIZE, stream_in)) {
    if (strlen(buf) == 0)
      continue;
    bufp = &(buf[0]);
    j = 0;
    while ((token = strsep(&bufp, ","))) {
      val = strtod(token, NULL);
      if (val) {
        matrix_in[i * size + j] = val;
        nnz[j]++;
      }
      j++;
    }
    i++;
  }
  fclose(stream_in);
}

template <typename T> void write_output_matrix_templ(T *matrix_out, long size,
  char *fn_out) {

  long i, j;
  FILE *stream_out;

  stream_out = fopen(fn_out, "w");
  for (i = 0; i < size; i++) {
    for (j = 0; j < size-1; j++) {
      fprintf(stream_out, "%e,", matrix_out[i * size + j]);
    }
    fprintf(stream_out, "%e\n", matrix_out[(i+1) * size - 1]);
  }
  fclose(stream_out);
}

extern "C" {
  void read_input_matrix_f(float *matrix_in, long *nnz, long size,
    char *fn_in) {
    read_input_matrix_templ<float>(matrix_in, nnz, size, fn_in);
  }
  void read_input_matrix_d(double *matrix_in, long *nnz, long size,
    char *fn_in) {
    read_input_matrix_templ<double>(matrix_in, nnz, size, fn_in);
  }
  void write_output_matrix_f(float *matrix_out, long size,
    char *fn_out) {
    write_output_matrix_templ<float>(matrix_out, size, fn_out);
  }
  void write_output_matrix_d(double *matrix_out, long size,
    char *fn_out) {
    write_output_matrix_templ<double>(matrix_out, size, fn_out);
  }
}
