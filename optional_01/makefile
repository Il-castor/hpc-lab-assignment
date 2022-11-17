
DATA ?= data/fvcorr.domn.097K
FILE ?= euler3d_cpu_offload.cpp
EXE = ${FILE:.cpp=}

all: ${EXE}

${EXE}: ${FILE}
	clang++ -O3 ${FILE} -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -o ${EXE} -lm

profile:
	sudo LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/ext/lib:${LD_LIBRARY_PATH} LIBRARY_PATH=/usr/ext/lib:${LIBRARY_PATH} /usr/local/cuda-10.0/bin/nvprof ./${EXE} ${DATA}

run: ${EXE}
	./${EXE} ${DATA}
	
clean:
	rm -rf euler3d_cpu_original euler3d_cpu_parallel euler3d_cpu_offloading ${EXE}
	rm -rf density density_energy momentum

.PHONY: all profile run clean
