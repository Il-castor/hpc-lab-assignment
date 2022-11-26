
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

BUILD_DIR ?= ./build

SRCS := ${FILE}
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o) $(EXERCISE:%=$(BUILD_DIR)/%.o)

all: ${EXE}

$(EXE): $(OBJS)
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCFLAGS) $(OBJS) -o $@ $(NVLDFLAGS)

$(BUILD_DIR)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

profile:
	sudo LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/ext/lib:${LD_LIBRARY_PATH} LIBRARY_PATH=/usr/ext/lib:${LIBRARY_PATH} /usr/local/cuda-10.0/bin/nvprof ./${EXE} ${DATA}

run: ${EXE}
	./${EXE} ${DATA}

clean:
	rm -rf euler3d_cpu_original euler3d_cpu_parallel euler3d_cpu_offloading ${EXE}
	rm -rf density density_energy momentum

.PHONY: all profile run clean
