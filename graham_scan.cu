#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stack>
#include <algorithm>

#include "kernels.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const int WIDTH = 800;
const int HEIGHT = 600;

void checkCUDAError(const char *);
void generatePointCloud(point *pts, int size, float bottomLX, float bottomLY, float squareSize);

float crossZ(point p1, point p2, point p3)
{
    return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
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

    // calculate the angle associated with each points vector from p0
    point unit_x;
    unit_x.x = 1;
    unit_x.y = 0;

    point pt;
    point v;
    float len_v;
    float cos_theta;
    for (i = 1; i < NUM_POINTS; ++i)
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
void generatePointCloud(point *pts, int size, float bottomLX, float bottomLY, float squareSize){
    for (int i = 0; i<size; ++i){
        pts[i].x = bottomLX + static_cast<float>(rand())/RAND_MAX * squareSize;
        pts[i].y = bottomLY + static_cast<float>(rand())/RAND_MAX * squareSize;
    }
}

// TODO: function for rendering point cloud with convex hull

int main(void)
{
    // TODO
    point pointsArray[NUM_POINTS];

    int size = NUM_POINTS;    // Size of the point cloud
    float bottomLeftX = 0.0f; // Bottom left corner of the square
    float bottomLeftY = 0.0f;
    float squareSize = 10.0f; // Size of the square containing the point cloud

    generatePointCloud(pointsArray, size, bottomLeftX, bottomLeftY, squareSize);

    std::stack<point> s = grahamScanCPU(pointsArray);
    point pt;
    while (!s.empty())
    {
        pt = s.top();
        s.pop();
        printf("stack point (%f, %f)", pt.x, pt.y);
    }
}