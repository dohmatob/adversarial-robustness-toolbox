import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def plot_multivariate_normal(mean, cov, x_width=5, y_width=5, nb_x=100,
                             nb_y=100, ax=None):
    """
    Plot multivariate normal pdf
    """
    asser len(mean) == len(cov) == len(cov[0])
    x1, x2 = np.meshgrid(np.linspace(-x_width + mean[0], x_width + mean[0],
                                     num=nb_x),
                         np.linspace(-y_width + mean[1], y_width + mean[1],
                                     num=nb_y))
    model = scipy.stats.multivariate_normal(mean, cov)
    z = np.ndarray(list(x1.shape) + [2])
    z[:, :, 0] = x1
    z[:, :, 1] = x2
    pdf = model.pdf(z)
    if ax is None:
        _, ax = plt.subplots()
    im = ax.contourf(x1, x2, pdf)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel("pdf")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    plt.tight_layout()
    return ax
