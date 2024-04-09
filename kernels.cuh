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

// __global__ void merge_basic_kernel(points *d_points)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;

//     if (tid < NUM_POINTS)
//     {
//         // Merge operation using bubble sort
//         for (int i = 0; i < NUM_POINTS - 1; i++)
//         {
//             for (int j = 0; j < NUM_POINTS - i - 1; j++)
//             {
//                 // Compare angles
//                 if (d_points->angle[j] > d_points->angle[j + 1])
//                 {
//                     // Swap
//                     point temp;
//                     temp.x = d_points->x[j];
//                     temp.y = d_points->y[j];
//                     temp.angle = d_points->angle[j];

//                     d_points->x[j] = d_points->x[j + 1];
//                     d_points->y[j] = d_points->y[j + 1];
//                     d_points->angle[j] = d_points->angle[j + 1];

//                     d_points->x[j + 1] = temp.x;
//                     d_points->y[j + 1] = temp.y;
//                     d_points->angle[j + 1] = temp.angle;
//                 }
//             }
//         }
//         __syncthreads();
//     }
// }

// __global__ void merge_tiled_kernel(int *A, int m, int *B, int n, int *C, int tile_size)
// { /* shared memory allocation */
//     extern __shared__ int shareAB[];
//     int *A_S = &shareAB[0];
//     int *B_S = &shareAB[tile_size];
//     // shareA is first half of shareAB // shareB is second half of shareAB
//     int C_curr = blockIdx.x * device_ceil((m + n) / gridDim.x); // start point of block's C subarray int C_next = min((blockIdx.x+1) * device_ceil((m+n)/gridDim.x), (m+n)); // ending point
//     if (threadIdx.x == 0)
//     {
//         A_S[0] = co_rank(C_curr, A, m, B, n); // Make block-level co-rank values visible co_rank (C_next, A, m, B, n); // to other threads in the block
//         A_S[1] = co_rank(C_next, A, m, B, n);
//     }
//     __syncthreads();
//     int A_curr = A S[0];
//     int A next = A_S[1];
//     int B_curr = C_curr int B next = C_next
//     syncthreads();
//     A_curr;
//     A_next;
// }

__device__ void movePointInPoints(points *pts, int dest, int src)
{
    pts->x[dest] = pts->x[src];
    pts->y[dest] = pts->y[src];
    pts->angle[dest] = pts->angle[src];
}

// __global__ void radixSort(points *data, int n, int bit)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n)
//     {
//         unsigned int *u_data = reinterpret_cast<unsigned int *>(&data->angle[idx]); // Treat float as unsigned int
//         if ((*u_data & (1 << bit)) == 0)
//         {
//             // Move '0' bit elements to the beginning
//             // Use exclusive prefix sum to find the starting index for '0' bit elements
//             // You can use thrust::exclusive_scan for better performance
//             int count0 = 0;
//             for (int i = 0; i < idx; ++i)
//             {
//                 unsigned int *u_i_data = reinterpret_cast<unsigned int *>(&data->angle[i]);
//                 if ((*u_i_data & (1 << bit)) == 0)
//                     count0++;
//             }
//             __syncthreads();
//             // data[count0] = data[idx];
//             movePointInPoints(data, count0, idx);
//         }
//         else
//         {
//             // Move '1' bit elements to the end
//             // Use exclusive prefix sum to find the starting index for '1' bit elements
//             // You can use thrust::exclusive_scan for better performance
//             int count1 = 0;
//             for (int i = n - 1; i > idx; --i)
//             {
//                 unsigned int *u_i_data = reinterpret_cast<unsigned int *>(&data->angle[i]);
//                 if ((*u_i_data & (1 << bit)) != 0)
//                     count1++;
//             }
//             __syncthreads();
//             // data[n - 1 - count1] = data[idx];
//             movePointInPoints(data, n - 1 - count1, idx);
//         }
//     }
// }

// __global__ void radix_sort_iter(unsigned int *input, unsigned int *output, unsigned int *bits, unsigned int N, unsigned int iter)
// {
//     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//     unsigned int key, bit;
//     if (i < N)
//     {
//         key = input[i];
//         bit(key >> iter) & 1;
//         bits[i] = bit;
//     }
//     exclusiveScan(bits, N);
//     if (i < N)
//     {
//         unsigned int numOnesBefore = bits[i];
//         unsigned int numOnesTotal = bits[N];
//         unsigned int dst = (bit == 0) ? (i - numOnesBefore) : (output[dst] = key);
//         output[dst] = key;
//     }
// }

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
