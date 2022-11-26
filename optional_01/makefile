
ifndef CUDA_HOME
CUDA_HOME:=/usr/local/cuda
endif

MKDIR_P = mkdir -p
DATA ?= data/fvcorr.domn.097K
FILE ?= euler3d_cpu_original.cpp
FILENAME = $(basename $(FILE))
EXE = ${FILENAME}

NVCC=$(CUDA_HOME)/bin/nvcc
CXX=g++

OPT := -O2 -g
NVOPT := -Xcompiler -fopenmp -lineinfo -arch=sm_53 --ptxas-options=-v --use_fast_math
CXXFLAGS := $(OPT) $(EXT_CXXFLAGS)
LDFLAGS := -lm -lcudart $(EXT_LDFLAGS)

NVCFLAGS:=$(CXXFLAGS) $(NVOPT)
NVLDFLAGS:=$(LDFLAGS) -lgomp

ifeq ($(FILENAME), euler3d_cpu_offload)
CXX=clang++
CXXOFLAGS := -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda
endif

all: ${EXE}

ifeq ($(suffix $(FILE)), .cu)
$(EXE): ${FILENAME}.cu
	$(NVCC) $(NVCFLAGS) $< -o ${EXE}
else
$(EXE): ${FILENAME}.cpp
	$(CXX) $(CXXFLAGS) $(CXXOFLAGS) $< -o ${EXE}
endif

profile:
	sudo LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/ext/lib:${LD_LIBRARY_PATH} LIBRARY_PATH=/usr/ext/lib:${LIBRARY_PATH} /usr/local/cuda-10.0/bin/nvprof ./${EXE} ${DATA}

run: ${EXE}
	./${EXE} ${DATA}

clean:
	rm -rf build/*
	rm -rf euler3d_cpu_original euler3d_cpu_parallel euler3d_cpu_offloading euler3d_cuda ${EXE}
	rm -rf density density_energy momentum

.PHONY: all profile run clean
