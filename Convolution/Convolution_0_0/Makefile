all: main.cpp StencilOps.o
	icc -O3 -std=c++14 -Wall -w -mavx512f -mfma -fopenmp main.cpp StencilOps.o

StencilOps.o: StencilOps.cpp
	icc -O3 -std=c++14 -Wall -w -mavx512f -mfma -fopenmp -fno-alias StencilOps.cpp -c

assembly: StencilOps.cpp
	icc -O3 -std=c++14 -Wall -w -mavx512f -mfma -fopenmp -fno-alias StencilOps.cpp -S

clean:
	rm a.out *.s *.o

