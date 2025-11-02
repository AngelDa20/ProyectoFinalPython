# ui/viz_matplotlib.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_robot(model, joints, ax=None, show=True, equal_axes=True, title=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    xs, ys, zs = joints[:,0], joints[:,1], joints[:,2]
    ax.plot(xs, ys, zs, marker='o')
    ax.scatter([xs[0]], [ys[0]], [zs[0]], s=40)   # base
    ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], s=40)  # tool tip

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title(title or model.name)

    if equal_axes:
        _set_axes_equal(ax)
    if show:
        plt.show()
    return ax

def _set_axes_equal(ax):
    bounds = np.array([ax.get_xbound(), ax.get_ybound(), ax.get_zbound()], dtype=float)
    minv = bounds[:,0].min()
    maxv = bounds[:,1].max()
    span = maxv - minv
    c = (minv + maxv) / 2.0
    r = span / 2.0 if span > 0 else 1.0
    ax.set_xlim([c - r, c + r])
    ax.set_ylim([c - r, c + r])
    ax.set_zlim([c - r, c + r])
