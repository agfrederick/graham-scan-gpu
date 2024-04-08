# BId:    cwalia
# Name:   Chantel Rose Walia
# Date:   02/24/2024
# Class:  CS680 HW2

import numpy as np
import matplotlib.pyplot as plt

# a) Write a Python function for computing the convex hull of a point cloud in 2-space. It has
# one input parameter: a column-major data matrix representing the point cloud. It returns an
# NumPy array: the column indices of the input that define the convex hull. These indices are
# sorted counterclockwise around the hull, starting with the lowest point.


def left_turn(p1, p2, p3):
    turn_type = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    if turn_type == 0:
        return 0
    return 1 if turn_type > 0 else -1


def angle(p1, p2):
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]
    magnitude = np.sqrt(x**2 + y**2)
    if np.all(magnitude == 0):
        return 0
    return x / magnitude


def convex_hull(p_cloud):
    stack = []
    size = p_cloud.shape[1]

    p0 = p_cloud[:, 0]
    for i in range(1, p_cloud.shape[1]):
        if p_cloud[1, i] < p0[1] or (p_cloud[1, i] == p0[1] and p_cloud[0, i] < p0[0]):
            p0 = p_cloud[:, i]

    temp_p = p_cloud[:, p_cloud[0] != p0[0]]
    temp_p = temp_p[:, np.argsort(angle(temp_p, p0))]
    
    stack.append(p0)
    if temp_p.shape[1] > 0:
        stack.append(temp_p[:, 0])

    for i in range(1, size - 1):
        while len(stack) > 1 and left_turn(stack[-2], stack[-1], temp_p[:, i]) != 1:
            stack.pop()
        stack.append(temp_p[:, i])

    hull_indices = []
    for i in stack:
        for j in range(p_cloud.shape[1]):
            if np.array_equal(i, p_cloud[:, j]):
                hull_indices.append(j)
                break

    return hull_indices


# (b) Write a Python function that renders a point cloud and its convex hull. It takes two input
# parameters, a column-major data matrix of the point cloud and an array of the column indices
# of its convex hull. Use matplotlib to render. (As a challenge, you could try implementing using
# pyglet.)


def render(p_cloud, hull_indices):
    points = p_cloud
    hull_points = [points[:, i] for i in hull_indices]

    for i in range(len(hull_points) - 1):
        plt.plot(
            [hull_points[i][0], hull_points[i + 1][0]],
            [hull_points[i][1], hull_points[i + 1][1]],
            color="cyan",
            linestyle="-",
        )

    plt.plot(
        [hull_points[-1][0], hull_points[0][0]],
        [hull_points[-1][1], hull_points[0][1]],
        color="cyan",
        linestyle="-",
    )

    plt.scatter(points[0], points[1], color="green", label="Point Cloud")

    plt.title("Convex Hull")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()


# (c) To test the two functions on random point clouds, write a Python function with three input
# parameters: the size of the point cloud, the bottom left corner of the square containing the point
# cloud, the size of the square. It should return a column-major data matrix representing a random
# point cloud of this size that lies inside this square.


def generate_point_cloud(pc_size, bottom_left_corner, sq_size):
    x_vals = np.round(
        np.random.uniform(
            bottom_left_corner[0], bottom_left_corner[0] + sq_size, pc_size
        ),
        decimals=2,
    )
    y_vals = np.round(
        np.random.uniform(
            bottom_left_corner[1], bottom_left_corner[1] + sq_size, pc_size
        ),
        decimals=2,
    )

    return np.vstack((x_vals, y_vals))


# (d) Use the three functions to test your code by drawing the convex hull of random point clouds.

if __name__ == "__main__":
    i = 0
    while i < 3:
        print(f"\nRender {i+1}/3\n")
        pc_size = int(input("Please enter point cloud size (int): "))
        if pc_size < 3:
            print("OOPS point cloud size must be >= 3\n")
            continue

        bottom_left_x = float(
            input(
                "Please enter the bottom left X-coordinate for the square containing the point cloud (int/float): "
            )
        )
        bottom_left_y = float(
            input(
                "Please enter the bottom left Y-coordinate for the square containing the point cloud (int/float): "
            )
        )
        sq_size = int(input("Please enter the size of the square (int): "))

        point_cloud = generate_point_cloud(
            pc_size, (bottom_left_x, bottom_left_y), sq_size
        )

        print(f"------------\nPoint cloud:\n{point_cloud}\n")
        hull_indices = convex_hull(point_cloud)

        print(f"hull_indeces: {hull_indices}\n")
        render(point_cloud, hull_indices)
        i += 1
