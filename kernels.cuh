#ifndef KERNEL_H
#define KERNEL_H

#define NUM_POINTS 32 // TODO: try different numbers of points
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

__global__ void lowestPoint_kernel(points *d_points, points *d_reduced_points)
{
    // find index of point with lowest y-val
    // if multiple, break tie with lowest x-val
    // Reference: lab3
    __shared__ point shared_points[NUM_POINTS];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int shared_idx = threadIdx.x;

    // load a point into shared memory
    point pt;
    pt.x = d_points->x[idx];
    pt.y = d_points->y[idx];
    shared_points[shared_idx] = pt;
    // printf("loading mark %f to shared mem index %d\n", record.assignment_mark, shared_idx);
    __syncthreads();

    // reduce
    int i = 2 * shared_idx;
    for (int stride = 1; stride < NUM_POINTS; stride *= 2)
    {
        if (shared_idx % stride == 0)
        {
            if (shared_points[i].y > shared_points[i + stride].y)
            {
                shared_points[i] = shared_points[i + stride];
            }
            else if (shared_points[i].y == shared_points[i + stride].y) // TODO: better float equality check
            {
                if (shared_points[i].x > shared_points[i + stride].x)
                {
                    shared_points[i] = shared_points[i + stride];
                }
            }
        }
        __syncthreads();
    }
    if (shared_idx == 0)
    {
        d_reduced_points->x[blockIdx.x] = shared_points[0].x;
        d_reduced_points->y[blockIdx.x] = shared_points[0].y;
    }
    __syncthreads();
}

__global__ void findCosAngles_kernel(points *d_points, point p0)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if not lowest point p0:
    // Compute vector
    // Find cosine of angle with x-axis, store in angle entry of struct
    // calculate the angle associated with each points vector from p0
    // point unit_x;
    // unit_x.x = 1;
    // unit_x.y = 0;

    // point pt;
    // point v;
    // float len_v;
    // float cos_theta;
    // if (pt.x != p0.x && pt.y != p0.y) // TODO: float equality
    // {
    //     pt.x = d_points->x[idx];
    //     pt.y = d_points->y[idx];
    //     v.x = pt.x - p0.x;
    //     v.y = pt.y = p0.y;
    //     len_v = pow((pow(v.x, 2) + pow(v.y, 2)), 0.5);
    //     cos_theta = (v.x * unit_x.x + v.y * unit_x.y) / len_v;
    //     d_points->angle[idx] = cos_theta;
    // }
}

__global__ void sorting_kernel(points *d_pts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO
}

#endif // KERNEL_H