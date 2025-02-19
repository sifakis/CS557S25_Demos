CUDA_ROOT = $(shell dirname `which nvcc`)/..

NVCC=nvcc
NVCC_FLAGS=--forward-unknown-to-host-compiler --expt-relaxed-constexpr --std=c++17 -lineinfo --gpu-architecture=sm_86 --resource-usage -Xptxas -v -uumn
CUDA_LIBRARY=-L$(CUDA_ROOT)/lib64
CUDA_LIBS=-lcudart -lcuda

ABI_FLAG =
CFLAGS = -std=c++17 -Wall -w -mavx2 -mfma -fopenmp
INCLUDES = -I./include -I$(CUDA_ROOT)/include
LIBRARY = $(CUDA_LIBRARY)
LIBS = $(CUDA_LIBS)
DEP = Makefile include/*.h


BUILD_DEBUG   = $(NVCC) $(NVCC_FLAGS) $(CFLAGS) $(ABI_FLAG) -O0 -g
BUILD_RELEASE = $(NVCC) $(NVCC_FLAGS) $(CFLAGS) $(ABI_FLAG) -DNDEBUG -O3

# Avoid associating these build target with files of the same name
.PHONY: all clean

all: release

def: release

.DEFAULT_GOAL := def

#########

release/main.o: $(DEP) src/main.cpp
	@mkdir -p release
	$(BUILD_RELEASE) $(INCLUDES) -o release/main.o -c src/main.cpp

release/Laplacian.o: $(DEP) src/Laplacian.cpp
	@mkdir -p release
	$(BUILD_RELEASE) $(INCLUDES) -o release/Laplacian.o -c src/Laplacian.cpp

release/Kernels.o: $(DEP) src/Kernels.cu
	@mkdir -p release
	$(BUILD_RELEASE) $(INCLUDES) $(CUDA_LIBRARY) -o release/Kernels.o -c src/Kernels.cu $(CUDA_LIBS)

release: release/main.o release/Kernels.o release/Laplacian.o
	$(BUILD_RELEASE) $(INCLUDES) $(LIBRARY) -o release/cudaTest release/main.o release/Kernels.o release/Laplacian.o $(LIBS)

#########

clean:
	@rm -rf release
