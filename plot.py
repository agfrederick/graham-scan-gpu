import matplotlib.pyplot as plt


def render(pts_file: str, hull_file: str, out_file: str) -> None:
    with open(pts_file) as f:
        pts = f.readlines()
    x_vals_pts = [float(pt.split(" ")[0]) for pt in pts if pt.strip()]
    y_vals_pts = [float(pt.split(" ")[1]) for pt in pts if pt.strip()]

    with open(hull_file) as f:
        hull = f.readlines()
    x_vals_hull = [float(pt.split(" ")[0]) for pt in hull if pt.strip()]
    y_vals_hull = [float(pt.split(" ")[1]) for pt in hull if pt.strip()]

    if len(x_vals_hull) > 0:
        if x_vals_hull[0] != x_vals_hull[-1] or y_vals_hull[0] != y_vals_hull[-1]:
            x_vals_hull.append(x_vals_hull[0])  
            y_vals_hull.append(y_vals_hull[0])  # connect first and last points

    plt.scatter(x_vals_pts, y_vals_pts, color="darkorchid")

    if out_filename == "outputs/cpu_plot.png":
        plt.plot(x_vals_hull, y_vals_hull, color="purple")
        plt.title("CPU Convex Hull")
        plt.xlabel("X point")
        plt.ylabel("Y point")
    else:

        plt.title("GPU Convex Hull")
        plt.plot(x_vals_hull, y_vals_hull, color="purple")
        plt.xlabel("X point")
        plt.ylabel("Y point")
        

    plt.savefig(out_file)
    plt.close()


if __name__ == "__main__":
    pts_filename = "cpu_points.txt"
    hull_filename = "cpu_stack.txt"
    out_filename = "outputs/cpu_plot.png"
    render(pts_filename, hull_filename, out_filename)

    pts_filename = "gpu_points.txt"
    hull_filename = "gpu_stack.txt"
    out_filename = "outputs/gpu_plot.png"
    render(pts_filename, hull_filename, out_filename)
