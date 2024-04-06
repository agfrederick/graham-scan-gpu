import matplotlib.pyplot as plt


def render(pts_file: str, hull_file: str, out_file: str) -> None:
    with open(pts_file) as f:
        pts = f.readlines()
        print(f"num points: {len(pts)}")
    x_vals_pts = [float(pt.split(" ")[0]) for pt in pts if pt != ""]
    y_vals_pts = [float(pt.split(" ")[1]) for pt in pts if pt != ""]

    with open(hull_file) as f:
        hull = f.readlines()
        print(f"num points: {len(pts)}")
    x_vals_hull = [float(pt.split(" ")[0]) for pt in hull if pt != ""]
    y_vals_hull = [float(pt.split(" ")[1]) for pt in hull if pt != ""]

    plt.scatter(x_vals_pts, y_vals_pts, color="darkorchid")
    plt.plot(x_vals_hull, y_vals_hull, color="purple")
    plt.plot(
        [x_vals_hull[0], x_vals_hull[-1]],
        [y_vals_hull[0], y_vals_hull[-1]],
        color="purple",
    )  # connect first and last points
    # plt.show()
    plt.savefig(out_file)


if __name__ == "__main__":
    # pts_filename = input("Enter filename for points: ")
    # hull_filename = input("\nEnter filename for hull: ")
    # out_filename = input("\nEnter filename for output: ")
    pts_filename = "outputs/cpu_points.txt"
    hull_filename = "outputs/cpu_stack.txt"
    out_filename = "outputs/plot.png"
    render(pts_filename, hull_filename, out_filename)
