#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stack>
#include <algorithm>
#include <cstdlib> // For rand()
#include <iostream>
#include <fstream>
#include <iomanip>

#include "kernels.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void mergeSortFRThisTime(int *arr, int begin, int end)
{
    printf("MergeSortFRThisTime(begin:%d,end:%d)\n", begin, end);
    printf("[");
    for (int i = begin; i <= end; i++)
    {
        printf("%d,", arr[i]);
    }
    printf("]\n");
    if (begin >= end)
        return;

    int mid = begin + (end - begin) / 2;
    printf("\tmid: %d\n", mid);
    mergeSortFRThisTime(arr, begin, mid);
    mergeSortFRThisTime(arr, mid + 1, end);

    int size = (end + 1 - begin);

    int leftArray[mid - begin + 1];
    int rightArray[end - mid];

    // Copy elements to leftArray
    for (int i = 0; i <= mid - begin; ++i)
    {
        leftArray[i] = arr[begin + i];
        printf("left arr[%d] = %d ", i, leftArray[i]);
    }
    printf("\n");

    // Copy elements to rightArray
    for (int i = 0; i <= end - mid; ++i)
    {
        rightArray[i] = arr[mid + 1 + i];
        printf("right arr[%d] = %d ", i - mid, rightArray[i - mid]);
    }
    printf("\n");

    int *leftArrayGPU;
    int *rightArrayGPU;
    int *testGPU;

    cudaMalloc((void **)&leftArrayGPU, (mid - begin + 1) * sizeof(int));
    cudaMalloc((void **)&rightArrayGPU, (end - mid) * sizeof(int));
    cudaMalloc((void **)&testGPU, size * sizeof(int));

    cudaMemcpy(leftArrayGPU, leftArray, (mid - begin + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(rightArrayGPU, rightArray, (end - mid) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(testGPU, arr, size * sizeof(int), cudaMemcpyHostToDevice);

    merge_basic_kernel<<<1, size>>>(leftArrayGPU, (mid - begin + 1), rightArrayGPU, (end - mid), testGPU);
    cudaDeviceSynchronize();

    cudaMemcpy(arr + begin, testGPU, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(leftArrayGPU);
    cudaFree(rightArrayGPU);
    cudaFree(testGPU);
}

int main(int argc, char **argv)
{

    int size = 10;
    int test[10] = {10, 9, 8, 7, 5, 6, 4, 3, 2, 1};

    mergeSortFRThisTime(test, 0, size - 1);

    for (int i = 0; i < size; ++i)
    {
        printf("%d\n", test[i]);
    }
}