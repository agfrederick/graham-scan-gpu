import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np


def convexHull(point_cloud: np.ndarray) -> list[int]:
    """
    point_cloud: np.ndarray of points shape (2, n) such that each column has the x, y coords of a point
    return: ordered list of the indices of each point in the convex hull in the point_cloud
    """

    num_points = point_cloud.shape[1]

    # find point p0 with min y
    min_pt_idx = 0
    for i in range(num_points):
        if point_cloud[:, i][1] < point_cloud[:, min_pt_idx][1]:
            min_pt_idx = i

    # break ties with same min y
    lowest_points = []
    for i in range(num_points):
        if i != min_pt_idx and floatEq(point_cloud[:, i][1], point_cloud[:, min_pt_idx][1]):
            print(point_cloud[:, i])
            lowest_points.append(i)
    if len(lowest_points) > 0:
        min_x = point_cloud[:, min_pt_idx][0]
        for idx in lowest_points:
            if point_cloud[:, idx][0] < min_x:
                min_x = point_cloud[:, idx][0]
                min_pt_idx = idx
    p0 = point_cloud[:, min_pt_idx]

    # build vectors and cos angles with p0
    pt_angles = []
    unit_x = np.array([1, 0]) # unit vector on positive x axis
    for i in range(num_points):
        if i != min_pt_idx:
            pt = point_cloud[:, i]
            v = pt - p0
            len_v = (v[0]**2 + v[1]**2)**0.5
            cos_theta = (v[0] * unit_x[0] + v[1] * unit_x[1]) / len_v
            pt_angles.append((pt, cos_theta))


    # sort points by the cosine angles
    sorted_pts = [p[0] for p in sorted(pt_angles, key=lambda x: x[1])]

    stack = deque()
    # push(p0, S); push(p1, S); push(p2, S)
    stack.append(p0)
    stack.append(sorted_pts[0])
    stack.append(sorted_pts[1])
    for j in range(2, num_points - 1): 
        pj = sorted_pts[j]
        top = stack.pop() 
        next_top = stack.pop() 
        stack.append(next_top)
        stack.append(top)
        cross_z = crossZ(pj, top, next_top)
        # while angle formed by nextToTop(S), top(S), and pj makes nonleft turn
        while cross_z < 0:
            stack.pop()
            top = stack.pop() 
            next_top = stack.pop() 
            stack.append(next_top)
            stack.append(top)
            cross_z = crossZ(pj, top, next_top)

        stack.append(pj)
    hull = list(stack)

    # find the indexes in the point cloud of the convex hull points
    idxs = []
    for pt in hull:
        for i in range(points.shape[1]):
            point = points[:, i]
            if floatEq(pt[0], point[0]) and floatEq(pt[1], point[1]):
                idxs.append(i)

    return idxs

def crossZ(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Return the z-coordinate of the cross product of two vectors formed by the points
    p1-p2 and p1-p3
    """
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

def render(point_cloud: np.ndarray, hull_idx: list[int]) -> None:
    """
    point_cloud: np.ndarray of points shape (2, n) such that each column has the x, y coords of a point
    hull_idx: ordered list of the indices of each point in the convex hull in the point_cloud
    Plots the point cloud and the convex hull
    """
    c_hull = [point_cloud[:, i] for i in hull_idx]
    x_vals_hull = [pt[0] for pt in c_hull]
    y_vals_hull = [pt[1] for pt in c_hull]
    x_vals = list(point_cloud[0, :])
    y_vals = list(point_cloud[1, :])
    plt.scatter(x_vals, y_vals, color="darkorchid")
    plt.plot(x_vals_hull, y_vals_hull, color="purple")
    plt.plot([x_vals_hull[0], x_vals_hull[-1]], [y_vals_hull[0], y_vals_hull[-1]], color="purple") # connect first and last points
    plt.show()

def genPointCloud(num_points: int, blc: tuple, size: float) -> np.ndarray:
    """
    num_points: int, number of points to generate in the point cloud
    blc: tuple of 2 float vals (x, y) representing coordinates of the bottom left corner of the square containing the point cloud
    size: float: the size of sides of the square containing the point cloud
    return: np.ndarray of points shape (2, n) such that each column has the x, y coords of a point
    """
    points = np.zeros((2, num_points))
    for i in range(num_points):
        points[0, i] = random.random() * size + blc[0] # x-coord
        points[1, i] = random.random() * size + blc[1] # y-coord
    return points

def floatEq(f1: float, f2: float) -> bool:
    """Test if two floats f1 and f2 are equal by seeing if their difference
    is below a certain margin eps.
    """
    eps = 0.0001
    if abs(f1-f2) > eps:
        return False
    return True

if __name__ == "__main__":
    points = genPointCloud(100, (10,50), 150.0)
    hull = convexHull(points)
    render(points, hull)
