#!/bin/bash

clang -O2 -g -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -I. -DPOLYBENCH_TIME -DPRINT_HASH -DLARGE_DATASET -Ipolybench durbin_gpu.c polybench/polybench.c -o durbin_gpu.out -lm
