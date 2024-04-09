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
// void mergeSort(int *test, int left, int right)
// {

//     int size = sizeof(test) / sizeof(test[0]);

//     int mid = size / 2;

//     int *leftArrayGPU;
//     int *rightArrayGPU;
//     int *testGPU;

//     int leftArray[mid];
//     int rightArray[size - mid];

//     if (left < right)
//     {
//         cudaMalloc((void **)&leftArrayGPU, mid * sizeof(int));
//         cudaMalloc((void **)&rightArrayGPU, (size - mid) * sizeof(int));
//         cudaMalloc((void **)&testGPU, size * sizeof(int));
//         // Copy elements to leftArray
//         for (int i = 0; i < mid; ++i)
//         {
//             leftArray[i] = test[i];
//         }

//         // Copy elements to rightArray
//         for (int i = mid; i < size; ++i)
//         {
//             rightArray[i - mid] = test[i];
//         }
//         cudaMemcpy(leftArrayGPU, leftArray, mid * sizeof(int), cudaMemcpyHostToDevice);
//         cudaMemcpy(rightArrayGPU, rightArray, (size - mid) * sizeof(int), cudaMemcpyHostToDevice);
//         // int A = test[]

//         cudaMemcpy(test, testGPU, (size) * sizeof(int), cudaMemcpyDeviceToHost);
//         int mid = left + (right - left) / 2;

//         // Sort first and second halves
//         mergeSort(test, left, mid);
//         mergeSort(test, mid + 1, right);

//         // Merge the sorted halves
//         merge_basic_kernel<<<1, size>>>(leftArrayGPU, 5, rightArrayGPU, 5, testGPU);
//         cudaDeviceSynchronize();
//         cudaMemcpy(test, testGPU, (size) * sizeof(int), cudaMemcpyDeviceToHost);
//         cudaFree(leftArrayGPU);
//         cudaFree(rightArrayGPU);
//         cudaFree(testGPU);

//     }
// }

// void mergeSort(int *input, int *output, int left, int right)
// {
//     if (left >= right)
//         return;

//     int mid = (left + right) / 2;

//     mergeSort(input, output, left, mid);
//     mergeSort(input, output, mid + 1, right);

//     // Merge operation executed on GPU
//     int size = right - left + 1;
//     int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     merge_basic_kernel<<<numBlocks, BLOCK_SIZE>>>(input, output, left, right);
// }

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

    // merge(array, begin, mid, end);
    int size = (end + 1 - begin);

    // int mid = size / 2;

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
    // point pointsArray[NUM_POINTS];
    // point pointsArray2[NUM_POINTS];

    // generatePointCloud(pointsArray, SIZE, BOTTOMLEFTX, BOTTOMLEFTY, SQUARESIZE);

    // // make copy of array to use in GPU function, original will be modified by CPU function
    // for (int i = 0; i < NUM_POINTS; ++i)
    // {
    //     pointsArray2[i] = pointsArray[i];
    // }

    // point pt;
    int size = 10;
    int test[10] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    int test_output[4]; // = {9, 8, 4, 5, 2, 7, 6, 3, 1, 2};
    // int size = sizeof(test) / sizeof(test[0]);

    // int mid = size / 2;

    // int *leftArrayGPU;
    // int *rightArrayGPU;
    // int *testGPU;

    // int leftArray[mid];
    // int rightArray[size - mid];

    // cudaMalloc((void **)&leftArrayGPU, mid * sizeof(int));
    // cudaMalloc((void **)&rightArrayGPU, (size - mid) * sizeof(int));
    // cudaMalloc((void **)&testGPU, size * sizeof(int));

    // // Copy elements to leftArray
    // for (int i = 0; i < mid; ++i)
    // {
    //     leftArray[i] = test[i];
    // }

    // // Copy elements to rightArray
    // for (int i = mid; i < size; ++i)
    // {
    //     rightArray[i - mid] = test[i];
    // }
    // cudaMemcpy(leftArrayGPU, leftArray, mid * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(rightArrayGPU, rightArray, (size - mid) * sizeof(int), cudaMemcpyHostToDevice);
    // // int A = test[]
    // merge_basic_kernel<<<1, 10>>>(leftArrayGPU, 5, rightArrayGPU, 5, testGPU);
    // cudaDeviceSynchronize();

    // cudaMemcpy(test1, testGPU, (size) * sizeof(int), cudaMemcpyDeviceToHost);
    // int size = sizeof(test) / sizeof(test[0]);
    // int *d_arr_input;
    // int *d_arr_output;
    // cudaMalloc(&d_arr_input, size * sizeof(int));
    // cudaMalloc(&d_arr_output, size * sizeof(int));
    // cudaMemcpy(d_arr_input, test, size * sizeof(int), cudaMemcpyHostToDevice);

    // mergeSort(d_arr_input, d_arr_output, 0, size - 1);

    // cudaMemcpy(test, d_arr_output, size * sizeof(int), cudaMemcpyDeviceToHost);

    // // Free device memory
    // cudaFree(d_arr_input);
    // cudaFree(d_arr_output);

    mergeSortFRThisTime(test, 0, size - 1);

    for (int i = 0; i < size; ++i)
    {
        printf("%d\n", test[i]);
    }

    // std::stack<point> s_cpu = grahamScanCPU(pointsArray);
    // writeToFile(pointsArray, NUM_POINTS, s_cpu, "cpu_points.txt", "cpu_stack.txt");

    // std::stack<point> s_gpu = grahamScanGPU(pointsArray2);
    // writeToFile(pointsArray2, NUM_POINTS, s_gpu, "gpu_points.txt", "gpu_stack.txt");
}