#!/usr/bin/python

import re
import sys

assert(len(sys.argv) == 2)
filename = sys.argv[1]

fd = open(filename, "r")
contents = fd.read()
fd.close()

res = re.search("0: Each of the (\d+) workers will solve", contents)
nWorker = int(res.group(1))

res = re.search("1: We have (\d+) thread\(s\)", contents)
nThreads = int(res.group(1))

size = 32768
density = 10

#print ("%d %d %d %d" % (nWorker, nThreads, size, density))

durationsWorker = dict()
durationsBcast = list()
durationsGatherv = list()
cpuTBuild = dict()
cpuTInv = dict()

results = re.finditer("0: Wall time elapsed for Bcast: (\d+)ms", contents)
for res in results:
    durationsBcast.append(int(res.group(1)))

results = re.finditer("0: Wall time elapsed for Gatherv: (\d+)ms", contents)
for res in results:
    durationsGatherv.append(int(res.group(1)))

repetitions = len(durationsBcast)
workerAt = dict()
for i in range(nWorker):
    workerAt[i] = 0
for i in range(repetitions):
    durationsWorker[i] = list()
    cpuTBuild[i] = list()
    cpuTInv[i] = list()

results = re.finditer("(\d+): Wall time elapsed: (\d+)ms", contents)
for res in results:
    w = int(res.group(1)) - 1
    #print("Worker %d, round %d" % (w, workerAt[w]))
    durationsWorker[workerAt[w]].append(int(res.group(2)))
    workerAt[w] += 1

for i in range(nWorker):
    workerAt[i] = 0
results = re.finditer("(\d+): CPU time sm build: (\d+)ms", contents)
for res in results:
    w = int(res.group(1)) - 1
    cpuTBuild[workerAt[w]].append(int(res.group(2)))
    workerAt[w] += 1

for i in range(nWorker):
    workerAt[i] = 0
results = re.finditer("(\d+): CPU time sm calc: (\d+)ms", contents)
for res in results:
    w = int(res.group(1)) - 1
    cpuTInv[workerAt[w]].append(int(res.group(2)))
    workerAt[w] += 1

bcastAvg = 0
gathervAvg = 0
fastestAvg = 0
slowestAvg = 0
sumWorkertimeAvg = 0
sumCpuTBuildAvg = 0
sumCpuTInvAvg = 0

# ignore results from first run due to it typically being an outlyer
for i in range(1, repetitions):
    bcast = durationsBcast[i]
    bcastAvg += bcast
    gatherv = durationsGatherv[i]
    gathervAvg += gatherv
    fastest = min(durationsWorker[i])
    fastestAvg += fastest
    slowest = max(durationsWorker[i])
    slowestAvg += slowest
    sumWorkertime = sum(durationsWorker[i])
    sumWorkertimeAvg += sumWorkertime
    sumCpuTBuild = sum(cpuTBuild[i])
    sumCpuTBuildAvg += sumCpuTBuild
    sumCpuTInv = sum(cpuTInv[i])
    sumCpuTInvAvg += sumCpuTInv
#    print ("%d %d %d %d %d %d %d %d %d %d %d" % (nWorker, nThreads, size, density, bcast, gatherv, fastest, slowest, sumWorkertime, sumCpuTBuild, sumCpuTInv))



bcastAvg /= (repetitions-1)
gathervAvg /= (repetitions-1)
fastestAvg /= (repetitions-1)
slowestAvg /= (repetitions-1)
sumWorkertimeAvg /= (repetitions-1)
sumCpuTBuildAvg /= (repetitions-1)
sumCpuTInvAvg /= (repetitions-1)
print ("%d %d %d %d %d %d %d %d %d %d %d" % (nWorker, nThreads, size, density, bcastAvg, gathervAvg, fastestAvg, slowestAvg, sumWorkertimeAvg, sumCpuTBuildAvg, sumCpuTInvAvg))
