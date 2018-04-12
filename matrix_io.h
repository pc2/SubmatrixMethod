#define BUF_SIZE 131072

#ifdef __cplusplus
extern "C" {
#endif

void read_input_matrix_f(float *matrix_in, long *nnz, long size, char *fn_in);
void read_input_matrix_d(double *matrix_in, long *nnz, long size, char *fn_in);
void write_output_matrix_f(float *matrix_out, long size, char *fn_out);
void write_output_matrix_d(double *matrix_out, long size, char *fn_out);

#ifdef __cplusplus
}
#endif
