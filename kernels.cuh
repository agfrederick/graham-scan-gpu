#ifndef KERNEL_H
#define KERNEL_H

#define NUM_POINTS 64 // TODO: try different numbers of points
#define THREADS_PER_BLOCK 64
#define NUM_BLOCKS (NUM_POINTS / THREADS_PER_BLOCK)
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
    else
    {
        d_points->angle[idx] = INFINITY;
    }
}

__device__ void movePointInPoints(points *pts, int dest, int src)
{
    pts->x[dest] = pts->x[src];
    pts->y[dest] = pts->y[src];
    pts->angle[dest] = pts->angle[src];
}

__device__ int device_ceil(float x)
{
    int ix = (int)x;
    return (x > ix) ? ix + 1 : ix;
}

__device__ void merge_sequential(int *A, int m, int *B, int n, int *C)
{
    printf("Merge sequential reached\n");
    for (int i = 0; i < m; i++)
    {
        printf("A[%d] = %d ", i, A[i]);
    }
    printf("\n");

    for (int i = 0; i < n; i++)
    {
        printf("B[%d] = %d ", i, B[i]);
    }
    printf("\n");
    int i = 0; // Index into A
    int j = 0; // Index into B
    int k = 0; // Index into C

    while ((i < m) && (j < n))
    {
        if (A[i] <= B[j])
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
    for (int i = 0; i < m + n; ++i)
    {
        printf("C[%d] = %d ", i, C[i]);
    }
    printf("\n");
}

__device__ int co_rank(int k, int *A, int m, int *B, int n)
{
    // printf("co_rank reached\n");
    int i = k < m ? k : m; // i = min (k, m)
    int j = k - i;
    int i_low = 0 > (k - n) ? 0 : k - n; // i_low = max (0, k-n) 0> (k-m) ? 0: k-m; // i_low = max (0, k-m)
    int j_low = 0 > (k - m) ? 0 : k - m;
    int delta;
    bool active = true;
    while (active)
    {
        // printf("active\n");
        if (i > 0 && j < n && A[i - 1] > B[j])
        {
            // printf("cond1\n");
            delta = ((i - i_low + 1) >> 1); // device_ceil((i - i_low) / 2);
            // printf("delta1 %d\n", delta);
            j_low = j;
            j = j + delta;
            i = i - delta;
        }
        else if (j > 0 && i < m && B[j - 1] >= A[i])
        {
            // printf("cond2\n");
            delta = ((j - j_low + 1) >> 1); // device_ceil((j - j_low) / 2);
            // printf("delta2 %d\n", delta);
            i_low = i;
            i = i + delta;
            j = j - delta;
        }
        else
        {
            // printf("setting inactive\n");
            active = false;
        }
    }
    // printf("co_rank returning %d\n", i);
    return i;
}

__global__ void merge_basic_kernel(int *A, int m, int *B, int n, int *C)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Original data passed n\n\t");

    for (int i = 0; i < m; i++)
    {
        printf("A[%d] = %d (%d) ", i, A[i], tid);
    }
    printf("\n");
    for (int i = 0; i < n; i++)
    {
        printf("B[%d] = %d (%d)", i, B[i], tid);
    }
    printf("\n");
    // printf("Basic kernel reached\n");
    int elementsPerThread = device_ceil((m + n) / blockDim.x * gridDim.x);
    int k_curr = tid * elementsPerThread;
    int k_next = min((tid + 1) * elementsPerThread, m + n);
    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;
    merge_sequential(&A[i_curr], i_next - i_curr, &B[j_curr], j_next - j_curr, &C[k_curr]);
}

#endif // KERNEL_H
