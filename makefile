build:
	mpicxx -fopenmp -c cFunctions.c -o cFunctions.o
	nvcc -I./inc -c cuda_histogram.cu -o cuda_histogram.o
	mpicxx -fopenmp -o mpiCudaOpemMP  cFunctions.o cuda_histogram.o  /usr/local/cuda-9.1/lib64/libcudart_static.a  -ldl -lrt

clean:
	rm -f *.o ./mpiCudaOpemMP

run:
	mpiexec -np 2 ./mpiCudaOpemMP < numbers.txt

runOn2:
	mpiexec -np 2 -machinefile  mf  -map-by  node  ./mpiCudaOpemMP


