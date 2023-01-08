CC=mpiicc
CFLAGS=-O2 -Wall -qopenmp -qmkl -mt_mpi
# CFLAGS=-g -O0 -Wall -qopenmp -qmkl -mt_mpi
LDFLAGS=$(CFLAGS)

BINARIES = mpi-matrix-inv matlab-to-csc csc-to-matlab mkl-matrix-inv

.PHONY: all clean

all: $(BINARIES)

mpi-matrix-inv: mpi-matrix-inv.o
	$(CC) $(LDFLAGS) -o $@ $^

mkl-matrix-inv: mkl-matrix-inv.o matrix_io.o timespec_subtract.o
	$(CC) $(LDFLAGS) -o $@ $^

matlab-to-csc: matlab-to-csc.o matrix_io.o
	$(CC) $(LDFLAGS) -o $@ $^

csc-to-matlab: csc-to-matlab.o matrix_io.o
	$(CC) $(LDFLAGS) -o $@ $^

clean:
	rm -f *.o $(BINARIES)
