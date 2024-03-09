#ifndef KERNEL_H
#define KERNEL_H

#define NUM_POINTS 2048 // TODO: try different numbers of points
#define THREADS_PER_BLOCK 256

#include <stdio.h>

struct point
{
    float x;
    float y;
    float angle;

    bool operator<(const point &pt) const
    {
        return angle < pt.angle;
    }
};

struct points
{
    float x[NUM_POINTS];
    float y[NUM_POINTS];
    float angle[NUM_POINTS];
};

__global__ void lowestPoint_kernel(points *pts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO:
    // find index of point with lowest y-val
    // if multiple, break tie with lowest x-val
    // Reference: lab3
}

__global__ void findCosAngles_kernel(points *d_pts, int lowest_idx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO if not lowest point:
    // Compute vector
    // Find cosine of angle with x-axis, store in angle entry of struct
}

__global__ void sorting_kernel(points *d_pts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
}

#endif // KERNEL_H