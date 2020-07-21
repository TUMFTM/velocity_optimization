from matplotlib import pyplot as plt
import numpy as np
try:
    import tikzplotlib
except ImportError:
    print('Warning: No module tikzplotlib found. Not necessary on car but for development.')

# Line width
LW = 1.5

# The higher the less bins are plotted
BIN_WIDTH = 20


class VisVP_Runtime:

    __slots__ = ()

    def __init__(self):
        """
        Python version: 3.5
        Created by: Thomas Herrmann (thomas.herrmann@tum.de)
        Created on: 01.02.2020

        Documentation: Class to plot velocity optimization's solver runtimes as histogram.
        """
        pass

    def plot_hist(self,
                  name_solver: str,
                  dt_solver: np.ndarray,
                  vis_options: dict):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.02.2020

        Documentation: Plots solver runtime histograms.

        Inputs:
        name_solver: the optimization solver's name (e.g., 'OSQP', 'qpOASES', 'IPOPT', ...)
        dt_solver: solver runtimes
        vis_options: user specified debug visualization options
        """

        plt.figure()

        cnt = plt.hist(dt_solver,
                       bins=int(np.ceil(len(dt_solver) / BIN_WIDTH)))[0]

        plt.xlabel('SQP (' + name_solver + ') runtime in ms')
        plt.ylabel('Count')

        plt.plot([np.mean(dt_solver), np.mean(dt_solver)],
                 [0, max(cnt)],
                 color='red', linestyle='--', linewidth=LW)

        plt.plot([np.median(dt_solver), np.median(dt_solver)],
                 [0, max(cnt)],
                 color='black', linestyle='-.', linewidth=LW)

        if vis_options['b_save_tikz']:
            tikzplotlib.save('SQP_' + name_solver + '_runtime.tex')

        plt.legend(['mean', 'median'])

        # --- Show histogram
        # plt.show()


if __name__ == '__main__':
    pass
