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

int SIZE = NUM_POINTS;    // Size of the point cloud
float BOTTOMLEFTX = 0.0f; // Bottom left corner of the square
float BOTTOMLEFTY = 0.0f;
float SQUARESIZE = 10.0f; // Size of the square containing the point cloud

void checkCUDAError(const char *);
void generatePointCloud(point *pts, int size, float bottomLX, float bottomLY, float squareSize);
point minPointGPU(points *h_points, points *h_points_result, points *d_points, points *d_points_result);
void calculateCosAnglesGPU(points *h_points, points *d_points, point p0);

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

        // assign values to struct arrays
        output->x[i] = x;
        output->y[i] = y;
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
        else if (pts[i].y == pts[min_pt_index].y) // TODO: better equality test for floats
        {
            if (pts[i].x < pts[min_pt_index].x || pts[i].x == pts[min_pt_index].x) // TODO: better equality test for floats
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
            v.y = pt.y = p0.y;
            len_v = pow((pow(v.x, 2) + pow(v.y, 2)), 0.5);
            cos_theta = (v.x * unit_x.x + v.y * unit_x.y) / len_v;
            pts[i].angle = cos_theta;
        }
    }
    for (int i = 0; i < NUM_POINTS; ++i)
    {
        // printf("pt angle CPU: (%f, %f) %f\n", pts[i].x, pts[i].y, pts[i].angle);
    }

    for (i = 0; i < NUM_POINTS; ++i)
    {
        bool min_pt_found = false;
        if (i == min_pt_index)
        {
            min_pt_found = true;
        }
        else if (min_pt_found)
        {
            pts[i - 1].x = pts[i].x;
            pts[i - 1].y = pts[i].y;
        }
    }

    // sort points by cos angle (using built in to start, maybe should make our own CPU sort?)
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
    for (int i = 0; i < size; ++i)
    {
        pts[i].x = bottomLX + static_cast<float>(rand()) / RAND_MAX * squareSize;
        pts[i].y = bottomLY + static_cast<float>(rand()) / RAND_MAX * squareSize;
        // printf("Rand pt: (%f, %f)\n", pts[i].x, pts[i].y);
    }
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

    // sort points using cosine angle and min point

    std::stack<point> s;

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
    int blocks = NUM_POINTS / THREADS_PER_BLOCK;
    dim3 numBlocks(blocks);
    dim3 threadsPerBlock(THREADS_PER_BLOCK);
    lowestPoint_kernel<<<numBlocks, threadsPerBlock>>>(d_points, d_points_result);
    cudaDeviceSynchronize();
    cudaMemcpy(h_points_result, d_points_result, sizeof(points), cudaMemcpyDeviceToHost);

    // Reduce the block level results on CPU
    for (int i = 0; i < blocks; ++i)
    {
        float x = h_points_result->x[i];
        float y = h_points_result->y[i];
        if (y < min_pt.y)
        {
            min_pt.x = x;
            min_pt.y = y;
        }
        else if (y == min_pt.y) // TODO: float comparison
        {
            if (x < min_pt.x)
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
    printf("GPU lowest point was found at %f, %f\n", min_pt.x, min_pt.y);
    printf("\tExecution time was %f ms\n", time);

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
    checkCUDAError("anlges: CUDA memcpy");

    cudaEventRecord(start, 0);
    int blocks = NUM_POINTS / THREADS_PER_BLOCK;
    dim3 numBlocks(blocks);
    dim3 threadsPerBlock(THREADS_PER_BLOCK);
    findCosAngles_kernel<<<numBlocks, threadsPerBlock>>>(d_points, p0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_points, d_points, sizeof(points), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    // output result
    printf("\tExecution time for angle finding was %f ms\n", time);

    for (int i = 0; i < NUM_POINTS; ++i)
    {
        // printf("pt angle GPU: (%f, %f) %f\n", h_points->x[i], h_points->y[i], h_points->angle[i]);
    }

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
        outFile1 << std::fixed << std::setprecision(2) << pts[i].x << " " << pts[i].y << std::endl;
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
        printf("Writing stack point (%f, %f)\n", pt.x, pt.y);
        outFile2 << std::fixed << std::setprecision(2) << pt.x << " " << pt.y << std::endl;
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

    point pt;
    std::stack<point> s_cpu = grahamScanCPU(pointsArray);
    writeToFile(pointsArray, NUM_POINTS, s_cpu, "cpu_points.txt", "cpu_stack.txt");
    // while (!s_cpu.empty())
    // {
    //     pt = s_cpu.top();
    //     s_cpu.pop();
    //     printf("CPU stack point (%f, %f)\n", pt.x, pt.y);
    // }

    std::stack<point> s_gpu = grahamScanGPU(pointsArray2);
    writeToFile(pointsArray2, NUM_POINTS, s_gpu, "gpu_points.txt", "gpu_stack.txt");

    // while (!s_gpu.empty())
    // {
    //     pt = s_gpu.top();
    //     s_gpu.pop();
    //     printf("GPU stack point (%f, %f)\n", pt.x, pt.y);
    // }
}