#!/usr/bin/env python3

# MIT License
# 
# Copyright (c) 2018 Paderborn Center for Parallel Computing
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import re
import sys
from numba import jit

basedir = "Matrix-data/"
noMolecules = int(sys.argv[1])
size = noMolecules * 3 * 2
basedir = "{}/data-{}/".format(basedir, noMolecules)

def load_matrix_from_file(fn, dim, threshold=0):
    A = np.zeros([dim, dim])
    parser = re.compile("^\s+(\d+)\s+(\d+)\s+(\S+)")

    with open(fn, "r") as fh:
        for line in fh:
            mtch = parser.match(line)
            if mtch:
                res = mtch.groups()
                if (A[int(res[0])-1][int(res[1])-1] != 0):
                    raise Exception("ALARM!")
                A[int(res[0])-1][int(res[1])-1] = float(res[2])

    return A

@jit
def build_submatrix(matrix_in, index):
    valuemask  = np.invert(matrix_in[index].mask)
    valuepos   = np.nonzero(valuemask)[0]
    submat_dim = len(valuepos)
    #print("\rBuilding submatrix no. {:4d} of dimension {:4d}".format(index, submat_dim), end="")
    submatrix  = np.zeros([submat_dim, submat_dim])
    for j in range(submat_dim):
        for k in range(submat_dim):
            if (matrix_in[valuepos[j]][valuepos[k]]):
                submatrix[j][k] = matrix_in[valuepos[j]][valuepos[k]]
    return submatrix, valuepos

def print_matrix_info(M):
    print("* {}x{} Matrix".format(np.shape(M)[0],np.shape(M)[1]))
    try:
        print("* Density: {}".format(1 - np.count_nonzero(M.mask)/(np.shape(M)[0]*np.shape(M)[1])))
    except AttributeError:
        print("* Density: {}".format(np.count_nonzero(M)/(np.shape(M)[0]*np.shape(M)[1])))
    if np.all(np.linalg.eigvals(M) > 0):
        print("* Matrix is positive definite")
    if np.all(M == M.T):
        print("* Matrix is symmetric")
    print("* Condition number: {}".format(np.linalg.cond(M)))
    print("* Spectral radius: {}".format(np.max(np.abs(np.linalg.eigvals(M)))))

fn = "{}/overlap.bin".format(basedir)
S = load_matrix_from_file(fn, size)
print_matrix_info(S)

fn = "{}/H_ortho.bin".format(basedir)
H_ortho = load_matrix_from_file(fn, size)

fn = "{}/hamiltonian.bin".format(basedir)
hamiltonian = load_matrix_from_file(fn, size)

fn = "{}/density_kernel.bin".format(basedir)
dkernel = load_matrix_from_file(fn, size)

S_inv = np.linalg.inv(S)
H_ortho_recal = hamiltonian.dot(S_inv)
np.linalg.norm(H_ortho_recal - H_ortho, 2)

band_energy = np.trace(dkernel.dot(hamiltonian))
print("Band energy: {}".format(band_energy))

band_energy_orth = np.trace(S.dot(dkernel).dot(H_ortho))
print("Band energy ortho: {}".format(band_energy_orth))
print("relError band energy ortho: {}".format(-np.abs(band_energy_orth - band_energy) / band_energy))

band_energy_orth_recal = np.trace(S.dot(dkernel).dot(H_ortho_recal))
print("Band energy ortho recalc: {}".format(band_energy_orth_recal))
print("relError band energy ortho recalc: {}".format(-np.abs(band_energy_orth_recal - band_energy) / band_energy))

S_masked = np.ma.masked_array(S, S == 0)
S_inv_approx = np.zeros([size,size])
for i in range(size):
    submatrix, indexes = build_submatrix(S_masked, i)
    submatrix_inv = np.linalg.inv(submatrix)
    for j in range(len(indexes)):
        S_inv_approx[indexes[j]][i] = submatrix_inv[j][np.where(indexes == i)[0][0]]
#print()

H_ortho_approx = hamiltonian.dot(S_inv_approx)
band_energy_orth_approx = np.trace(S.dot(dkernel).dot(H_ortho_approx))
print("Band energy ortho approx: {}".format(band_energy_orth_approx))
print("relError band energy ortho approx: {}".format(-np.abs(band_energy_orth_approx - band_energy) / band_energy))

band_energy_orth_approx_rough = np.trace(S.dot(dkernel).dot(hamiltonian))
print("Band energy ortho=I: {}".format(band_energy_orth_approx_rough))
print("relError band energy ortho=I: {}".format(-np.abs(band_energy_orth_approx_rough - band_energy) / band_energy))
