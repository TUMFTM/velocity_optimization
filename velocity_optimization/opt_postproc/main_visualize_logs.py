import os
import sys
import numpy as np

# custom modules
mod_local_trajectory_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(mod_local_trajectory_path)
from velocity_optimization.opt_postproc.vis.VisBenchmarkLogs import VisVP_Logs


if __name__ == "__main__":
    """
    Python version: 3.5
    Created by: Thomas Herrmann (thomas.herrmann@tum.de)
    Created on: 01.02.2020

    Documentation: This function visualizes calculated velocity from the SQP-planner including its constraints.
    """

    csv_name = mod_local_trajectory_path + '/logs/vp_sqp/sqp_emerg_2020_04_23_10_33.log'
    csv_name_ltpl = mod_local_trajectory_path + '/logs/ltpl/2020_04_09/14_13_12_data.csv'

    # Number of velocity points
    m = 50
    # ID of used velocity planner 'PerfSQP' or 'EmergSQP'
    sid = 'EmergSQP'

    # Number of log lines spanning one data block
    log_lines = 4

    # visualize all logs consecutively?
    b_movie = False

    # re-calculate QP from log-input?
    b_run_OSQP = True

    # run qpOASES solver?
    b_calc_qpOASES = True

    # do global plot of states for entire log?
    b_global_plot = True
    glob_lim = np.inf

    # calculate solution from log input using NLP solver IPOPT?
    b_calc_IPOPT = True

    # plot immediately or only solver data replay?
    b_immediate_plot_update = True

    # show plot of solver runtimes?
    b_calc_time_plot = True

    # save plots as tikz files?
    b_save_tikz = True

    vis_options = {'b_movie': b_movie,
                   'b_run_OSQP': b_run_OSQP,
                   'b_calc_qpOASES': b_calc_qpOASES,
                   'b_global_plot': b_global_plot,
                   'glob_lim': glob_lim,
                   'b_calc_IPOPT': b_calc_IPOPT,
                   'b_immediate_plot_update': b_immediate_plot_update,
                   'b_calc_time_plot': b_calc_time_plot,
                   'b_save_tikz': b_save_tikz}

    # --- Set up visualization object
    rL = VisVP_Logs(csv_name=csv_name,
                    csv_name_ltpl=csv_name_ltpl,
                    m=m,
                    sid=sid,
                    log_lines=log_lines,
                    vis_options=vis_options)

    # --- Start GUI
    rL.vis_log(int(0))
