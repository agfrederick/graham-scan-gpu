#ifndef KERNEL_H
#define KERNEL_H

#define NUM_POINTS 32 // TODO: try different numbers of points
#define THREADS_PER_BLOCK 32
#define NUM_BLOCKS (NUM_POINTS/THREADS_PER_BLOCK)
#define EPSILON 1e-6f // Example value, adjust as needed


#include <stdio.h>
#include <math.h>

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
    __shared__ point shared_points[NUM_POINTS];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < NUM_POINTS)
    {
        // Each thread loads a point into shared memory
        point pt;
        pt.x = d_points->x[idx];
        pt.y = d_points->y[idx];
        shared_points[threadIdx.x] = pt;
        __syncthreads();

        // Reduce 
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            // printf("Block %d: (%f, %f)\n", blockIdx.x, shared_points[threadIdx.x].x, shared_points[threadIdx.x].y);

            if (threadIdx.x < stride)
            {
                if (shared_points[threadIdx.x].y > shared_points[threadIdx.x + stride].y ||
                    (fabs(shared_points[threadIdx.x].y - shared_points[threadIdx.x + stride].y) < EPSILON &&
                    shared_points[threadIdx.x].x > shared_points[threadIdx.x + stride].x))
                {
                    shared_points[threadIdx.x] = shared_points[threadIdx.x + stride];
                }
            }
            // __syncthreads();
        }
            __syncthreads();

        // Store the minimum point of each block to output array
        if (threadIdx.x == 0)
        {
            // printf("Block %d: (%f, %f)\n", blockIdx.x, shared_points[0].x, shared_points[0].y);
            d_reduced_points->x[blockIdx.x] = shared_points[0].x;
            d_reduced_points->y[blockIdx.x] = shared_points[0].y;
        }
    }
}


__global__ void findCosAngles_kernel(points *d_points, point p0)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    point unit_x;
    unit_x.x = 1;
    unit_x.y = 0;

    point pt;
    point v;
    float len_v;
    float cos_theta;
    pt.x = d_points->x[idx];
    pt.y = d_points->y[idx];
    if (fabs(pt.x - p0.x) > EPSILON && fabs(pt.y - p0.y) > EPSILON)
    {
        v.x = pt.x - p0.x;
        v.y = pt.y - p0.y;
        len_v = sqrt((v.x * v.x + v.y * v.y));
        cos_theta = (v.x * unit_x.x + v.y * unit_x.y) / len_v;
        d_points->angle[idx] = cos_theta;
    }
}


__global__ void sorting_kernel(points *d_points)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // sort solely by angle in point struct
    if (idx < NUM_POINTS) {

        for (int i = 0; i < NUM_POINTS - 1; i++) {

            for (int j = 0; j < NUM_POINTS - i - 1; j++) {

                if (d_points->angle[j] > d_points->angle[j+1]) {

                    // selection sort

                    point temp;

                    temp.x = d_points->x[j];
                    temp.y = d_points->y[j];
                    temp.angle = d_points->angle[j];

                    d_points->x[j] = d_points->x[j + 1];
                    d_points->y[j] = d_points->y[j + 1];
                    d_points->angle[j] = d_points->angle[j + 1];

                    d_points->x[j + 1] = temp.x;
                    d_points->y[j + 1] = temp.y;
                    d_points->angle[j + 1] = temp.angle;

                    __syncthreads();

                }

            }
        }
    }
    
}


#endif // KERNEL_H
