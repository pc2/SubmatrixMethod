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
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define PATHLEN 255

struct properties {
  int size;
  int density;
  int condition;
};

MKL_INT find_elem(MKL_INT needle, MKL_INT *haystack, MKL_INT size) {
  MKL_INT l, r, c;
  l = 0;
  r = size;
  do {
    c = (l+r)/2;
    if (haystack[c] == needle) {
      return c;
    }
    if (haystack[c] < needle) {
      l = c+1;
    } else {
      r = c;
    }
  } while (l != r);
  return -1;
}

lapack_int invert_matrix(double *matrix, lapack_int size) {
  // First we need to compute the LU factorization using ?getrf
  lapack_int *ipiv, ret;
  ipiv = (lapack_int*) mkl_calloc(size, sizeof(lapack_int), 64);

  ret = LAPACKE_dgetrf(LAPACK_COL_MAJOR, size, size, matrix, size, ipiv);
  if (ret) {
    mkl_free(ipiv);
    return ret;
  }

  // And now we calculate the inverse using the LU factorization
  ret = LAPACKE_dgetri(LAPACK_COL_MAJOR, size, matrix, size, ipiv);
  mkl_free(ipiv);
  return ret;
}

void print_matrix(double *matrix, MKL_INT size) {
  MKL_INT i, j;
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      printf("%.2f\t", matrix[i * size + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void invert_submatrix(double *values, MKL_INT *row_ind, MKL_INT *col_ptr,
                      double *values_inv, int i, double *locDurBuild, double *locDurCalc) {

  MKL_INT nnz, k, l, kcal, lcal, idx;
  lapack_int ret;
  double *submatrix;
  double tStart, tEnd;

  nnz = col_ptr[i+1] - col_ptr[i];
  submatrix = (double*) mkl_calloc(nnz*nnz, sizeof(double), 64);

  tStart = omp_get_wtime();
  for (k = 0; k < nnz; k++) {
    for (l = 0; l < nnz; l++) {
      kcal = row_ind[col_ptr[i]+k];
      lcal = row_ind[col_ptr[i]+l];
      // We now have to copy M[kcal][lcal] to submatrix[k][l]
      // How to access M[kcal][lcal]? Calculate idx
      idx = find_elem(kcal, &(row_ind[col_ptr[lcal]]),
                      col_ptr[lcal+1]-col_ptr[lcal]);
      if (idx != -1) {
        submatrix[k*nnz+l] = values[col_ptr[lcal] + idx];
      }
    }
  }
  tEnd = omp_get_wtime();
  *locDurBuild = (tEnd - tStart);

  tStart = omp_get_wtime();
  ret = invert_matrix(submatrix, nnz);
  tEnd = omp_get_wtime();
  *locDurCalc = (tEnd - tStart);
  if (ret) {
    fprintf(stderr, "Inverting submatrix failed\n");
  }

//  tStart = omp_get_wtime();
  memcpy(values_inv,
         &(submatrix[find_elem(i, &(row_ind[col_ptr[i]]), nnz) * nnz]),
         nnz*sizeof(double));
//  tEnd = omp_get_wtime();
//  *locDurBuild += (tEnd - tStart);

  mkl_free(submatrix);
}

int main(int argc, char* argv[]) {

  int threadsupport;
  MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &threadsupport);
  if (threadsupport < MPI_THREAD_FUNNELED) {
    fprintf(stderr, "Could not initialize thread support.");
    omp_set_num_threads(1);
  }

  struct properties prop;
  int fd, world_rank, world_size, *displs, *recvcounts, mkl_threads;
  char fn_in_val[PATHLEN], fn_in_ri[PATHLEN], fn_in_cp[PATHLEN],
       fn_out_val[PATHLEN];
  MKL_INT *col_ptr, *row_ind, total_nnz, i, submatrices_per_worker, total_elem,
          my_first_col, next_first_col, submatrices_for_me;
  double *values, *values_inv, tStart, tEnd;
  FILE *fp;

  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // printf("%d: I'm alive\n", world_rank);

  if (world_rank == 0) {
    
    
    
/**************
 * MPI Rank 0 *
 **************/

    // We are the boot process. Decide on what to do, broadcast and gather.
    if (argc != 4) {
      fprintf(stderr,
        "%d: Main process needs to be called with parameters size density "
        "condition\n", world_rank);

      // printf("%d: Shutting down workers...\n", world_rank);
      prop.size = 0;
      MPI_Bcast(&prop, 3, MPI_INT, 0, MPI_COMM_WORLD);

      exit(EXIT_FAILURE);
    }
    prop.size = strtol(argv[1], NULL, 10);
    prop.density = strtol(argv[2], NULL, 10);
    prop.condition = strtol(argv[3], NULL, 10);

    submatrices_per_worker = prop.size / (world_size-1);
    printf("%d: Each of the %d workers will solve %d submatrices.\n",
      world_rank, (world_size-1), submatrices_per_worker);
    if (prop.size % (world_size-1) != 0) {
      fprintf(stderr, "%d: WARNING: Load imbalanced. Last worker will have to "
              "solve %d additional submatrices\n", world_rank,
              prop.size % (world_size-1));
    }


/* Main evaluation loop */

    int evalRep, evalChoice;
    
    for (evalRep = 0; evalRep < 5; evalRep++) {
      for (evalChoice = 1; evalChoice > 0; evalChoice--) {

        snprintf(fn_in_cp, PATHLEN, "sprandsym-s%d-d%d-c%d-n%d.cp", prop.size,
                 prop.density, prop.condition, evalChoice);
        snprintf(fn_out_val, PATHLEN, "sprandsym-s%d-d%d-c%d-n%d.inv.val",
                 prop.size, prop.density, prop.condition, evalChoice);
        snprintf(fn_in_val, PATHLEN, "sprandsym-s%d-d%d-c%d-n%d.val", prop.size,
                 prop.density, prop.condition, evalChoice);
        snprintf(fn_in_ri, PATHLEN, "sprandsym-s%d-d%d-c%d-n%d.ri", prop.size,
                 prop.density, prop.condition, evalChoice);
        snprintf(fn_out_val, PATHLEN, "sprandsym-s%d-d%d-c%d-n%d.out.val", prop.size,
                 prop.density, prop.condition, evalChoice);


        col_ptr = (MKL_INT*) calloc(prop.size+1, sizeof(MKL_INT));
        fp = fopen(fn_in_cp, "rb");
        fread(col_ptr, sizeof(MKL_INT), prop.size+1, fp);
        fclose(fp);
        
        total_nnz = col_ptr[prop.size];
        row_ind = (MKL_INT*) calloc(total_nnz, sizeof(MKL_INT));
        fp = fopen(fn_in_ri, "rb");
        fread(row_ind, sizeof(MKL_INT), total_nnz, fp);
        fclose(fp);
        
        values = (double*) calloc(total_nnz, sizeof(double));
        fp = fopen(fn_in_val, "rb");
        fread(values, sizeof(double), total_nnz, fp);
        fclose(fp);
    
/*      fp = fopen(fn_out_val, "wb");
        fseek(fp, total_nnz*sizeof(double)-1, SEEK_SET);
        fputc('\0', fp);
        fclose(fp);
        fd = open(fn_out_val, O_RDWR);

        values_inv = (double*) mmap(NULL, total_nnz*sizeof(double),
                                    PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
*/
        values_inv = (double*) calloc(total_nnz, sizeof(double));

        displs = (int*) calloc(world_size, sizeof(int));
        recvcounts = (int*) calloc(world_size, sizeof(int));
        for (i = 1; i < world_size; i++) {
          displs[i] = col_ptr[(i-1)*submatrices_per_worker];
          recvcounts[i] = col_ptr[i*submatrices_per_worker] -
                          col_ptr[(i-1)*submatrices_per_worker];
        }
        // Allow last worker to send all remaining results
        recvcounts[world_size-1] = col_ptr[prop.size] -
                                   col_ptr[(world_size-2) * submatrices_per_worker];


        tStart = MPI_Wtime();
        // printf("%d: Broadcasting information to all workers...\n", world_rank);
        MPI_Bcast(&prop, 3, MPI_INT, 0, MPI_COMM_WORLD);
        // printf("%d: ... done\n", world_rank);

#ifndef USE_BEEGFS
        // Send data to all workers
        MPI_Bcast(col_ptr, prop.size+1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(row_ind, total_nnz, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(values, total_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
        tEnd = MPI_Wtime();

        printf("%d: Wall time elapsed for Bcast: %dms\n", world_rank,
               (int)((tEnd-tStart)*1000));
      

        tStart = MPI_Wtime();
        // printf("%d: Waiting for results...\n", world_rank);
        MPI_Gatherv(NULL, 0, MPI_DOUBLE, values_inv, recvcounts, displs,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // printf("%d: ... done\n", world_rank);
        tEnd = MPI_Wtime();

        printf("%d: Wall time elapsed for Gatherv: %dms\n", world_rank,
               (int)((tEnd-tStart)*1000));
       
        fp = fopen(fn_out_val, "wb");
        fwrite(values_inv, sizeof(double), total_nnz, fp);
        fclose(fp);
        
        free(row_ind);
        free(values);
        free(recvcounts);
        free(displs);
        free(col_ptr);
        free(values_inv);
  //    munmap(values_inv, total_nnz*sizeof(double));


      }
    }

/* End of main evaluation loop */

    // printf("%d: Shutting down workers...\n", world_rank);
    prop.size = 0;
    MPI_Bcast(&prop, 3, MPI_INT, 0, MPI_COMM_WORLD);


  } else {





/***************
 * Worker code *
 ***************/

    // We are one of the workers. Run in a loop and wait for jobs.
    while (1) {
      printf("%d: Waiting for matrix properties...\n", world_rank);
      MPI_Bcast(&prop, 3, MPI_INT, 0, MPI_COMM_WORLD);
      printf("%d: ... received\n", world_rank);
      if (prop.size == 0) {
        printf("%d: Received signal to halt.\n", world_rank);
        break;
      }
      
      snprintf(fn_in_cp, PATHLEN, "sprandsym-s%d-d%d-c%d-n1.cp", prop.size,
               prop.density, prop.condition);
      snprintf(fn_out_val, PATHLEN, "sprandsym-s%d-d%d-c%d-n1.inv.val",
               prop.size, prop.density, prop.condition);
      snprintf(fn_in_val, PATHLEN, "sprandsym-s%d-d%d-c%d-n1.val", prop.size,
               prop.density, prop.condition);
      snprintf(fn_in_ri, PATHLEN, "sprandsym-s%d-d%d-c%d-n1.ri", prop.size,
               prop.density, prop.condition);
      
#ifdef USE_BEEGFS
# ifdef USE_MMAP
      fd = open(fn_in_cp, O_RDONLY);
      col_ptr = (MKL_INT*) mmap(NULL, (prop.size+1)*sizeof(MKL_INT), PROT_READ,
                                MAP_SHARED, fd, 0);
      close(fd);
# else
      fp = fopen(fn_in_cp, "rb");
      col_ptr = (MKL_INT*) calloc(prop.size+1, sizeof(MKL_INT));
      fread(col_ptr, sizeof(MKL_INT), prop.size+1, fp);
      fclose(fp);
# endif
#else
      col_ptr = (MKL_INT*) calloc(prop.size+1, sizeof(MKL_INT));
      MPI_Bcast(col_ptr, prop.size+1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
      
      total_nnz = col_ptr[prop.size];
#ifdef USE_BEEGFS
# ifdef USE_MMAP
      fd = open(fn_in_ri, O_RDONLY);
      row_ind = (MKL_INT*) mmap(NULL, total_nnz*sizeof(MKL_INT), PROT_READ,
                                MAP_SHARED, fd, 0);
      close(fd);
      fd = open(fn_in_val, O_RDONLY);
      values = (double*) mmap(NULL, total_nnz*sizeof(double), PROT_READ,
                              MAP_SHARED, fd, 0);
      close(fd);
# else
      row_ind = (MKL_INT*) calloc(total_nnz, sizeof(MKL_INT));
      values = (double*) calloc(total_nnz, sizeof(double));
      fp = fopen(fn_in_ri, "rb");
      fread(row_ind, sizeof(MKL_INT), total_nnz, fp);
      fclose(fp);
      fp = fopen(fn_in_val, "rb");
      fread(values, sizeof(double), total_nnz, fp);
      fclose(fp);
# endif
#else
      row_ind = (MKL_INT*) calloc(total_nnz, sizeof(MKL_INT));
      values = (double*) calloc(total_nnz, sizeof(double));
      MPI_Bcast(row_ind, total_nnz, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(values, total_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

      submatrices_per_worker = prop.size / (world_size-1);
      my_first_col = (world_rank-1)*submatrices_per_worker;
      if (world_rank == world_size-1) {
        // We are the last worker. Maybe we have to do additional work...
        submatrices_for_me = submatrices_per_worker +
                             (prop.size % (world_size-1));
        next_first_col = prop.size;
        
      } else {
        submatrices_for_me = submatrices_per_worker;
        next_first_col = world_rank*submatrices_per_worker;
      }
      total_elem = col_ptr[next_first_col] - col_ptr[my_first_col];
      values_inv = (double*) calloc(total_elem, sizeof(double));

      /* Optimize threading: We should do as much submatrices as possible in
       * parallel. If threads are left, leave them for MKL's internal
       * parallelism. */
      mkl_threads = omp_get_max_threads() / submatrices_for_me;
      if (mkl_threads < 1) {
        mkl_threads = 1;
      }
      mkl_set_num_threads(mkl_threads);

      printf("%d: We have %d thread(s) to solve %d submatrices. Give %d "
             "thread(s) to MKL for each submatrix operation.\n", world_rank,
             omp_get_max_threads(), submatrices_for_me, mkl_threads);

      double durationBuild = .0;
      double durationCalc = .0;
      tStart = MPI_Wtime();
      // printf("%d: Starting the number crunching\n", world_rank);
      #pragma omp parallel for schedule(dynamic) reduction(+:durationBuild,durationCalc)
      for (i = 0; i < submatrices_for_me; i++) {
        double locDurBuild, locDurCalc;
        // printf("%d: Inverting submatrix %d in thread %d.\n", world_rank,
        //        (world_rank-1)*submatrices_for_me + i, omp_get_thread_num());
        invert_submatrix(values, row_ind, col_ptr,
          &(values_inv[
            col_ptr[my_first_col + i] -
            col_ptr[my_first_col]
          ]), my_first_col + i, &locDurBuild, &locDurCalc);
        durationBuild += locDurBuild;
        durationCalc += locDurCalc;
      }
      tEnd = MPI_Wtime();

      printf("%d: Wall time elapsed: %dms\n", world_rank,
            (int)((tEnd-tStart)*1000));
      printf("%d: CPU time sm build: %dms\n", world_rank,
            (int)(durationBuild*1000));
      printf("%d: CPU time sm calc: %dms\n", world_rank,
            (int)(durationCalc*1000));


      // printf("%d: Send results to root\n", world_rank);
      MPI_Gatherv(values_inv, total_elem, MPI_DOUBLE, NULL, NULL, NULL,
                  MPI_DOUBLE, 0, MPI_COMM_WORLD);
      // printf("%d: ... done\n", world_rank);

      memset(values_inv, 0, total_elem * sizeof(double));
      free(values_inv);
#if defined USE_BEEGFS && defined USE_MMAP
      munmap(values, total_nnz*sizeof(double));
      munmap(row_ind, total_nnz*sizeof(MKL_INT));
      munmap(col_ptr, (prop.size+1)*sizeof(MKL_INT));
#else
      memset(values, 0, total_nnz*sizeof(double));
      memset(row_ind, 0, total_nnz*sizeof(MKL_INT));
      memset(col_ptr, 0, (prop.size+1)*sizeof(MKL_INT));
      free(values);
      free(row_ind);
      free(col_ptr);
#endif
    }

  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  exit(EXIT_SUCCESS);
}
