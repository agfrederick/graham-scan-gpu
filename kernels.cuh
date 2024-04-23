#ifndef KERNEL_H
#define KERNEL_H

#define NUM_POINTS 1024
#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS (NUM_POINTS / THREADS_PER_BLOCK)
#define EPSILON 1e-6f // Example value, adjust as needed
// #define tile_size 64

#include <stdio.h>
#include <math.h>

struct __align__(8) point
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
            if (threadIdx.x < stride)
            {
                if (shared_points[threadIdx.x].y > shared_points[threadIdx.x + stride].y ||
                    (fabs(shared_points[threadIdx.x].y - shared_points[threadIdx.x + stride].y) < EPSILON &&
                     shared_points[threadIdx.x].x > shared_points[threadIdx.x + stride].x))
                {
                    shared_points[threadIdx.x] = shared_points[threadIdx.x + stride];
                }
            }
        }
        __syncthreads();

        // Store the minimum point of each block to output array
        if (threadIdx.x == 0)
        {
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
    else
    {
        d_points->angle[idx] = 2.0f;
    }
}

__device__ int device_ceil(float x)
{
    int ix = (int)x;
    return (x > ix) ? ix + 1 : ix;
}

__device__ void merge_sequential(point *A, int m, point *B, int n, point *C)
{
    int i = 0;
    int j = 0;
    int k = 0;

    while ((i < m) && (j < n))
    {
        if (A[i].angle <= B[j].angle)
        {
            C[k++] = A[i++];
        }
        else
        {
            C[k++] = B[j++];
        }
    }
    if (i == m)
    {
        while (j < n)
        {
            C[k++] = B[j++];
        }
    }
    else
    {
        while (i < m)
        {
            C[k++] = A[i++];
        }
    }
}

__device__ int co_rank(int k, point *A, int m, point *B, int n)
{
    int i = k < m ? k : m; // i = min (k, m)
    int j = k - i;
    int i_low = 0 > (k - n) ? 0 : k - n;
    int j_low = 0 > (k - m) ? 0 : k - m;
    int delta;
    bool active = true;
    while (active)
    {
        if (i > 0 && j < n && A[i - 1].angle > B[j].angle)
        {
            delta = ((i - i_low + 1) >> 1);
            j_low = j;
            j = j + delta;
            i = i - delta;
        }
        else if (j > 0 && i < m && B[j - 1].angle >= A[i].angle)
        {
            delta = ((j - j_low + 1) >> 1);
            i_low = i;
            i = i + delta;
            j = j - delta;
        }
        else
        {
            active = false;
        }
    }
    return i;
}

__global__ void merge_basic_kernel(point *A, int m, point *B, int n, point *C)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int elementsPerThread = device_ceil((m + n) / blockDim.x * gridDim.x);
    int k_curr = tid * elementsPerThread;
    int k_next = min((tid + 1) * elementsPerThread, m + n);
    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;
    merge_sequential(&A[i_curr], i_next - i_curr, &B[j_curr], j_next - j_curr, &C[k_curr]);
}

// __global__ void merge_tiled_kernel(int *A, int m, int *B, int n, int *C)
// {
//     extern __shared__ int shareAB[];

//     int *A_s = &shareAB[0];
//     int *B_s = &shareAB[tile_size];
//     int C_curr = blockIdx.x * device_ceil((m + n) / gridDim.x);
//     int C_next = min((blockIdx.x + 1) * device_ceil((m + n) / gridDim.x), (m + n));

//     if (threadIdx.x == 0)
//     {
//         A_s[0] = co_rank(C_curr, A, m, B, n);
//         A_s[1] = co_rank(C_curr, A, m, B, n);
//     }

//     __syncthreads();

//     int A_curr = A_s[0];
//     int A_next = A_s[1];
//     int B_curr = C_curr - A_curr;
//     int B_next = C_next - A_next;

//     __syncthreads();

//     int counter = 0;
//     int C_length = C_next - C_curr;
//     int A_length = A_next - A_curr;
//     int B_length = B_next - B_curr;
//     int total_iteration = device_ceil((C_length) / tile_size);
//     int C_completed = 0;
//     int A_consumed = 0;
//     int B_consumed = 0;

//     while (counter < total_iteration)
//     {
//         for (int i = 0; i < tile_size; i += blockDim.x)
//         {
//             if (i + threadIdx.x < A_length - A_consumed)
//             {
//                 A_s[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
//             }
//         }

//         for (int i = 0; i < tile_size; i += blockDim.x)
//         {
//             if (i + threadIdx.x < A_length - A_consumed)
//             {
//                 B_s[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
//             }
//         }

//         __syncthreads();

//         int c_curr = threadIdx.x * (tile_size / blockDim.x);
//         int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);

//         c_curr = (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;

//         c_next = (c_next <= C_length - C_completed) ? c_next : C_length - C_completed;

//         int a_curr = co_rank(c_curr, A_s, min(tile_size, A_length - A_consumed), B_s, min(tile_size, B_length - B_consumed));
//         int b_curr = c_curr - a_curr;
//         int a_next = co_rank(c_next, A_s, min(tile_size, A_length - A_consumed), B_s, min(tile_size, B_length - B_consumed));
//         int b_next = c_next - a_next;

//         merge_sequential(A_s + a_curr, a_next - a_curr, B_s + b_curr, b_next - b_curr, C + C_curr + C_completed + c_curr);

//         counter++;
//         C_completed += tile_size;
//         A_consumed += co_rank(tile_size, A_s, tile_size, B_s, tile_size);
//         B_consumed = C_completed - A_consumed;

//         __syncthreads();
//     }
// }

#endif // KERNEL_H
