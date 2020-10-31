#pragma once
#define MAX_LINE 64
#define HISTOGRAM_SIZE 256


int* cuda_Task(int* arr, int size);
int* openMP_Task(int* src_arr,int size);
int* calculateHistogram(int *arr, unsigned int size);
void mergeTasks(int* dest, int* src);
char *readInput();
void printArr(int* arr, int size);
int *get_nums(char *buffer, int *size);
