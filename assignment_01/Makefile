ifndef CUDA_HOME
CUDA_HOME:=/usr/local/cuda
endif

FILE ?= durbin_originale.c

BUILD_DIR ?= ./build

NVCC=$(CUDA_HOME)/bin/nvcc
CXX=gcc

OPT:=-O2 -g
NVOPT:=-Xcompiler -fopenmp -lineinfo -arch=sm_53 --ptxas-options=-v --use_fast_math

CXXFLAGS:=$(OPT) -I. -Ipolybench $(EXT_CXXFLAGS)
LDFLAGS:=-lm -lcudart $(EXT_LDFLAGS)

NVCFLAGS:=$(CXXFLAGS) $(NVOPT)
NVLDFLAGS:=$(LDFLAGS) -lgomp

SRCS:= polybench/polybench.c
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o) $(FILE:%=$(BUILD_DIR)/%.o)
EXE=$(basename $(FILE))

$(EXE):	$(OBJS)
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCFLAGS) $(OBJS) -o $@ $(NVLDFLAGS)

$(BUILD_DIR)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.c.o: %.c
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

all: $(EXE)

.PHONY: run profile clean
run: $(EXE)
	./$(EXE)

profile: $(EXE)
	sudo LD_LIBRARY_PATH=$(CUDA_HOME)/lib:/usr/ext/lib:${LD_LIBRARY_PATH} LIBRARY_PATH=/usr/ext/lib:${LIBRARY_PATH} nvprof ./$(EXE)

clean:
	echo HELLO $(EXE)
	-rm -fr $(BUILD_DIR) *.exe *.out *~

MKDIR_P ?= mkdir -p
