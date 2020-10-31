#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "myProto.h"

enum ranks{MASTER,SLAVE,MAX_PROCS};

int main(int argc, char* argv[])
{
	int num_procs, my_rank;
	int size, *hist1,*hist2;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	if (num_procs != MAX_PROCS)
	{
		printf("Program requires %d proccess\n", MAX_PROCS);
		MPI_Abort(MPI_COMM_WORLD, 0);
	}

	if (my_rank == MASTER)
	{
		int *initial;
		char *buffer;
		buffer = readInput();
		initial = get_nums(buffer, &size);
		int received[HISTOGRAM_SIZE] = {0}; 	
		 
		if (initial == NULL){
			MPI_Abort(MPI_COMM_WORLD, 0);	
		}

		printf("Initial array:\n");
		printArr(initial, size);		

		//Send size of array, and half of the array
		MPI_Send(&size, 1, MPI_INT, SLAVE, 0, MPI_COMM_WORLD);
		int hist_size = size/2;
		MPI_Send(initial + size/2 ,ceil(((double)size)/2), MPI_INT, SLAVE, 0, MPI_COMM_WORLD);
		
		//Calculate half of array with CUDA and OpenMP
		hist1 = cuda_Task(initial, hist_size/2);
		hist2 = openMP_Task(initial+hist_size/2,ceil(((double)hist_size)/2));

		//Merge calculations to one histogram
		mergeTasks(hist1, hist2);	

		//Receive SLAVE's calculations and merge to final histogram
		MPI_Recv(received, HISTOGRAM_SIZE, MPI_INT, SLAVE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		mergeTasks(hist1, received);
		
		printf("\nHistogram Array:\n");
		printArr(hist1, HISTOGRAM_SIZE);
		free(initial);
		free(buffer);
	}
	else
	{
		//Receive array and size of array from MASTER 
		MPI_Recv(&size, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		int part_size = ceil(((double)size)/2);
		int* received = (int*)calloc(part_size,sizeof(int));
		
		MPI_Recv(received, part_size, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);	

		//Calculate histograms
		hist1 = cuda_Task(received, part_size/2);		
		hist2 = openMP_Task(received+(part_size/2),ceil(((double)part_size)/2));

		//Merge to one histogram
		mergeTasks(hist1, hist2);
		MPI_Send(hist1, HISTOGRAM_SIZE, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}

int* cuda_Task(int* arr, int size)
{
	return calculateHistogram(arr, size);
}


int* openMP_Task(int* arr,int size)
{
	int* histogram = (int*)calloc(HISTOGRAM_SIZE,sizeof(int));
	int* tmp_hist;

	#pragma omp parallel
	{
		const int tid = omp_get_thread_num();
		const int nthreads = omp_get_num_threads();

		//Put zero's in temp histogram
		#pragma omp single
		{
			tmp_hist = (int*)calloc(HISTOGRAM_SIZE*nthreads,sizeof(int));
			if(tmp_hist == NULL){
				printf("Calloc FAILED");
			}
		}

		//calculate histogram - every thread is in charge of part of arr 
		#pragma omp for
		for (int i = 0; i < size; i++) 
		{	
			tmp_hist[tid*HISTOGRAM_SIZE + arr[i]]++;
		}

		//Merge all temp histogram to one short histogram
		#pragma omp for
		for (int i = 0; i < HISTOGRAM_SIZE; i++)
		{
			for (int j = 0; j < nthreads; j++)
			{
				histogram[i] += tmp_hist[j*HISTOGRAM_SIZE + i]; 
			}
		}
	}
	free(tmp_hist);
	return histogram;
}


void mergeTasks(int* dest, int* src)
{
	#pragma omp for
	for (int i = 0; i < HISTOGRAM_SIZE; i++)
		dest[i] += src[i]; //Each thread merges specific cell in src to dst
}

char *readInput()
{
	//Enter input - at the end press ENTER twice to stop reading
	char temp[MAX_LINE];
	char *buffer = (char *)calloc(1, sizeof(char));
	int i = 1;

	while (scanf("%[^\n]%*c", temp) == 1)
	{
		buffer = (char *)realloc(buffer, MAX_LINE * i * sizeof(char));
		if (buffer == NULL)
		{
			printf("Error buffer malloc failed.\n");
			return NULL;
		}
		strcat(buffer, temp);
		strcat(buffer, "\n");
		i++;
	}

	return buffer;
}

void printArr(int* arr, int size)
{
	printf("\n=============================\n");
	for (int i = 0; i < size; i++)
	{
		if (i%10 == 0) printf("\n");
		printf("arr[%-3d] = %3d ", i, arr[i]);
	}
	printf("\n=============================\n");
	fflush(stdout);
}

int *get_nums(char *buffer, int *size)
{
	const char *delimiters = " \t\n";
	char *temp;
	temp = strtok(buffer, delimiters);
	*size = atoi(temp);
	int *numbers = (int *)malloc(sizeof(int) * (*size));
	if (numbers == NULL)
	{
        	printf("Error numbers malloc failed.\n");
        	return NULL;
	}

	temp = strtok(NULL, delimiters);
	for(int i=0; i < *size; i++){
		
		if (temp == NULL && i < (*size)-1){
			printf("Not enough numbers!\n");
			return NULL;
		}
		numbers[i] = atoi(temp);
        	temp = strtok(NULL, delimiters);
	}

	if (temp != NULL){
		printf("Too many numbers!\n");
		return NULL;
	}

    	return numbers;
}


