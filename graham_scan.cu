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

const int WIDTH = 800;
const int HEIGHT = 600;

int SIZE = NUM_POINTS;    // Size of the point cloud
float BOTTOMLEFTX = 0.0f; // Bottom left corner of the square
float BOTTOMLEFTY = 0.0f;
float SQUARESIZE = 10.0f; // Size of the square containing the point cloud

void checkCUDAError(const char *);
void generatePointCloud(point *pts, int size, float bottomLX, float bottomLY, float squareSize);
point minPointGPU(points *h_points, points *h_points_result, points *d_points, points *d_points_result);
void calculateCosAnglesGPU(points *h_points, points *d_points, point p0);
points sortPointsByAngleGPU(points *h_points, points *d_points, point p0);

float crossZ(point p1, point p2, point p3)
{
    return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
}

void pointArrayToPoints(point *pts, points *output)
{
    for (int i = 0; i < NUM_POINTS; ++i)
    {
        // get values from array of structs
        float x = pts[i].x;
        float y = pts[i].y;
        float angle = pts[i].angle;

        // assign values to struct arrays
        output->x[i] = x;
        output->y[i] = y;
        output->angle[i] = angle;
    }
}

std::stack<point> grahamScanCPU(point *pts)
{
    int i;
    int min_pt_index = 0;

    // Find index of minimum point
    for (i = 1; i < NUM_POINTS; ++i)
    {
        if (pts[i].y < pts[min_pt_index].y)
        {
            min_pt_index = i;
        }
        else if (fabs(pts[i].y - pts[min_pt_index].y) < EPSILON) // Check for y equality with epsilon
        {
            if (pts[i].x < pts[min_pt_index].x || fabs(pts[i].x - pts[min_pt_index].x) < EPSILON) // Check for x equality with epsilon
            {
                min_pt_index = i;
            }
        }
    }

    // assign minimum point p0
    point p0;
    p0.x = pts[min_pt_index].x;
    p0.y = pts[min_pt_index].y;

    printf("CPU lowest point was found at (%f, %f), index %d\n", p0.x, p0.y, min_pt_index);

    // calculate the angle associated with each points vector from p0
    point unit_x;
    unit_x.x = 1;
    unit_x.y = 0;

    point pt;
    point v;
    float len_v;
    float cos_theta;
    for (i = 0; i < NUM_POINTS; ++i)
    {
        if (i != min_pt_index)
        {
            pt.x = pts[i].x;
            pt.y = pts[i].y;
            v.x = pt.x - p0.x;
            v.y = pt.y - p0.y;
            len_v = pow((pow(v.x, 2) + pow(v.y, 2)), 0.5);
            cos_theta = (v.x * unit_x.x + v.y * unit_x.y) / len_v;
            pts[i].angle = cos_theta;
        }
    }
    bool min_pt_found = false;
    for (i = 0; i < NUM_POINTS; ++i)
    {
        if (i == min_pt_index)
        {
            min_pt_found = true;
        }
        else if (min_pt_found) // never true?
        {
            pts[i - 1].x = pts[i].x;
            pts[i - 1].y = pts[i].y;
            pts[i - 1].angle = pts[i].angle;
        }
    }
    pts[NUM_POINTS - 1].x = p0.x;
    pts[NUM_POINTS - 1].y = p0.y;
    pts[NUM_POINTS - 1].angle = 0.0;

    // sort points by cos angle
    std::sort(pts, pts + NUM_POINTS - 1); // ignoring last point, is no longer relevant after shift

    std::stack<point> s;
    s.push(p0);
    s.push(pts[0]);
    s.push(pts[1]);

    for (int j = 2; j < NUM_POINTS - 1; ++j)
    {
        point pj = pts[j];
        point top = s.top();
        s.pop();
        point next_top = s.top();
        s.pop();
        s.push(next_top);
        s.push(top);
        float cross_z = crossZ(pj, top, next_top);
        while (cross_z < 0)
        {
            s.pop();
            point top = s.top();
            s.pop();
            point next_top = s.top();
            s.pop();
            s.push(next_top);
            s.push(top);
            cross_z = crossZ(pj, top, next_top);
        }
        s.push(pj);
    }
    return s;
}

// function for generating random point cloud
// Generates an array of type point
void generatePointCloud(point *pts, int size, float bottomLX, float bottomLY, float squareSize)
{
    for (int i = 0; i < size; i++)
    {
        pts[i].x = bottomLX + static_cast<float>(rand()) / RAND_MAX * squareSize;
        pts[i].y = bottomLY + static_cast<float>(rand()) / RAND_MAX * squareSize;
        // printf("Rand pt: (%f, %f)\n", pts[i].x, pts[i].y);
    }
}

void pointsArrayToPoint(points *pts, point *output)
{
    for (int i = 0; i < NUM_POINTS; ++i)
    {
        output[i].x = pts->x[i];
        output[i].y = pts->y[i];
        output[i].angle = pts->angle[i];
    }
}

void mergeSortFRThisTime(point *arr, int begin, int end)
{
    if (begin >= end)
        return;

    int mid = begin + (end - begin) / 2;
    // printf("\tmid: %d\n", mid);
    mergeSortFRThisTime(arr, begin, mid);
    mergeSortFRThisTime(arr, mid + 1, end);

    int size = (end + 1 - begin);

    point testG[size];
    point leftArray[mid - begin + 1];
    point rightArray[end - mid];

    // Copy elements to leftArray
    for (int i = 0; i <= mid - begin; ++i)
    {
        leftArray[i] = arr[begin + i];
    }

    // Copy elements to rightArray
    for (int i = 0; i <= end - mid; ++i)
    {
        rightArray[i] = arr[mid + 1 + i];
    }

    point *leftArrayGPU;
    point *rightArrayGPU;
    point *testGPU;

    cudaMalloc((void **)&leftArrayGPU, (mid - begin + 1) * sizeof(point));
    cudaMalloc((void **)&rightArrayGPU, (end - mid) * sizeof(point));
    cudaMalloc((void **)&testGPU, size * sizeof(point));

    cudaMemcpy(leftArrayGPU, leftArray, (mid - begin + 1) * sizeof(point), cudaMemcpyHostToDevice);
    cudaMemcpy(rightArrayGPU, rightArray, (end - mid) * sizeof(point), cudaMemcpyHostToDevice);
    cudaMemcpy(testGPU, arr, size * sizeof(point), cudaMemcpyHostToDevice);

    merge_basic_kernel<<<NUM_BLOCKS, size>>>(leftArrayGPU, (mid - begin + 1), rightArrayGPU, (end - mid), testGPU);
    cudaDeviceSynchronize();

    cudaMemcpy(arr + begin, testGPU, size * sizeof(point), cudaMemcpyDeviceToHost);

    cudaFree(leftArrayGPU);
    cudaFree(rightArrayGPU);
    cudaFree(testGPU);
}

std::stack<point> grahamScanGPU(point *pts)
{
    points *h_points;
    points *h_points_result;
    points *d_points;
    points *d_points_result;

    h_points = (points *)malloc(sizeof(points));
    h_points_result = (points *)malloc(sizeof(points));
    cudaMalloc((void **)&d_points, sizeof(points));
    cudaMalloc((void **)&d_points_result, sizeof(points));

    pointArrayToPoints(pts, h_points);

    // Find minimum point
    point p0 = minPointGPU(h_points, h_points_result, d_points, d_points_result);

    // calculate cos angle with p0 for each point
    calculateCosAnglesGPU(h_points, d_points, p0);

    point *h_pts = (point *)malloc(NUM_POINTS * sizeof(point));

    std::stack<point> s;

    pointsArrayToPoint(h_points, h_pts);

    mergeSortFRThisTime(h_pts, 0, NUM_POINTS - 1);

    s.push(p0);
    point p1;
    p1.x = h_pts[0].x;
    p1.y = h_pts[0].y;
    p1.angle = h_pts[0].angle;
    s.push(p1);
    point p2;
    p2.x = h_pts[1].x;
    p2.y = h_pts[1].y;
    p2.angle = h_pts[1].angle;
    s.push(p2);

    for (int j = 2; j < NUM_POINTS - 1; ++j)
    {
        point pj;

        pj.x = h_pts[j].x;
        pj.y = h_pts[j].y;
        pj.angle = h_pts[j].angle;

        point top = s.top();
        s.pop();
        point next_top = s.top();
        s.pop();
        s.push(next_top);
        s.push(top);
        float cross_z = crossZ(pj, top, next_top);
        while (cross_z < 0.0)
        {
            s.pop();
            point top = s.top();
            s.pop();
            point next_top = s.top();
            s.pop();
            s.push(next_top);
            s.push(top);
            cross_z = crossZ(pj, top, next_top);
        }
        s.push(pj);
    }

    return s;
}

point minPointGPU(points *h_points, points *h_points_result, points *d_points, points *d_points_result)
{
    unsigned int i;
    point min_pt;
    float time;
    cudaEvent_t start, stop;

    float max;
    if (BOTTOMLEFTX < BOTTOMLEFTY)
    {
        max = BOTTOMLEFTY + SIZE;
    }
    else
    {
        max = BOTTOMLEFTX + SIZE;
    }

    min_pt.x = max;
    min_pt.y = max;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // memory copy records to device
    cudaMemcpy(d_points, h_points, sizeof(points), cudaMemcpyHostToDevice);
    checkCUDAError("Min point: CUDA memcpy");

    cudaEventRecord(start, 0);

    dim3 numBlocks(NUM_BLOCKS);
    dim3 threadsPerBlock(THREADS_PER_BLOCK);

    lowestPoint_kernel<<<numBlocks, threadsPerBlock>>>(d_points, d_points_result);
    checkCUDAError("Min point: CUDA kernel");

    cudaDeviceSynchronize();
    checkCUDAError("Min point: CUDA dev sync");

    cudaMemcpy(h_points_result, d_points_result, sizeof(points), cudaMemcpyDeviceToHost);
    checkCUDAError("Min point: CUDA memcpy back");

    // Reduce the block level results on CPU
    for (int i = 0; i < NUM_BLOCKS; ++i)
    {
        float x = h_points_result->x[i];
        float y = h_points_result->y[i];
        if (y < min_pt.y)
        {
            min_pt.x = x;
            min_pt.y = y;
        }
        else if (fabs(y - min_pt.y) < EPSILON) // Compare with epsilon
        {
            if (x < min_pt.x || fabs(x - min_pt.x) < EPSILON) // Compare with epsilon
            {
                min_pt.x = x;
                min_pt.y = y;
            }
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    // output result
    printf("GPU lowest point was found at (%f, %f)\n", min_pt.x, min_pt.y);
    printf("\tExecution time for lowest point was %f ms\n", time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return min_pt;
}

void calculateCosAnglesGPU(points *h_points, points *d_points, point p0)
{
    unsigned int i;
    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // memory copy records to device
    cudaMemcpy(d_points, h_points, sizeof(points), cudaMemcpyHostToDevice);
    checkCUDAError("angles: CUDA memcpy forward");

    cudaEventRecord(start, 0);

    dim3 numBlocks(NUM_BLOCKS);
    dim3 threadsPerBlock(THREADS_PER_BLOCK);

    findCosAngles_kernel<<<numBlocks, threadsPerBlock>>>(d_points, p0);
    checkCUDAError("angles: CUDA kernel");

    cudaDeviceSynchronize();
    checkCUDAError("angles: CUDA dev sync");

    cudaMemcpy(h_points, d_points, sizeof(points), cudaMemcpyDeviceToHost);
    checkCUDAError("angles: CUDA memcpy back");

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    // output result
    printf("\tExecution time for angle finding was %f ms\n", time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void writeToFile(point *pts, int num_points, std::stack<point> s, const std::string &filename_pts, const std::string &filename_stack)
{
    std::ofstream outFile1(filename_pts.c_str());
    if (!outFile1.is_open())
    {
        std::cerr << "Error opening file " << filename_pts << std::endl;
        return;
    }

    for (int i = 0; i < num_points; ++i)
    {
        outFile1 << std::fixed << std::setprecision(2) << pts[i].x << " " << pts[i].y << " " << pts[i].angle << std::endl;
    }

    outFile1.close();

    std::ofstream outFile2(filename_stack.c_str());
    if (!outFile2.is_open())
    {
        std::cerr << "Error opening file " << filename_stack << std::endl;
        return;
    }

    point pt;
    while (!s.empty())
    {
        pt = s.top();
        s.pop();
        // printf("Writing stack point (%f, %f)\n", pt.x, pt.y);
        outFile2 << std::fixed << std::setprecision(2) << pt.x << " " << pt.y << " " << pt.angle << std::endl;
    }

    outFile2.close();

    std::cout << "Points and stack written to " << filename_pts << " " << filename_stack << std::endl;
}

int main(int argc, char **argv)
{

    point pointsArray[NUM_POINTS];
    point pointsArray2[NUM_POINTS];

    generatePointCloud(pointsArray, SIZE, BOTTOMLEFTX, BOTTOMLEFTY, SQUARESIZE);

    // make copy of array to use in GPU function, original will be modified by CPU function
    for (int i = 0; i < NUM_POINTS; ++i)
    {
        pointsArray2[i] = pointsArray[i];
    }

    std::stack<point> s_cpu = grahamScanCPU(pointsArray);
    writeToFile(pointsArray, NUM_POINTS, s_cpu, "cpu_points.txt", "cpu_stack.txt");

    std::stack<point> s_gpu = grahamScanGPU(pointsArray2);
    writeToFile(pointsArray2, NUM_POINTS, s_gpu, "gpu_points.txt", "gpu_stack.txt");
}