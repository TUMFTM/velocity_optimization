import os
import sys
import numpy as np
import linecache
import json

# custom modules
vel_opt_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(vel_opt_path)
from velocity_optimization.opt_postproc.vis.VisBenchmarkLogs import VisVP_Logs


if __name__ == "__main__":
    """
    Python version: 3.5
    Created by: Thomas Herrmann (thomas.herrmann@tum.de)
    Created on: 01.02.2020
    Modified by: Tobias Klotz

    Documentation: This function visualizes calculated velocity from the SQP-planner including its constraints.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # USER INPUT -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    csv_name = vel_opt_path + '/logs/sqp_perf_2020_07_03_16_12.log'

    csv_name_ltpl = vel_opt_path + '/logs/sqp_perf_2020_06_08_09_15.log'

    params_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/params/'
    input_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/inputs/'

    # Number of log lines spanning one data block
    log_lines = 4

    # visualize all logs consecutively?
    b_movie = False
    # re-calculate QP from log-input?
    b_run_OSQP = True  # TODO: combine with solver options below

    # run qpOASES solver?
    b_calc_qpOASES = False  # TODO: combine with solver options below

    # Constant(True)/Variable(False) Power
    b_con_power = False

    # Choose Starting Idx of Log-File
    b_idx = 0
    # Plot Race Course with planning horizon
    # b_plot_course = True
    # Select Legend Item (Model, Solver, Friction, Alpha)
    b_vis_model_name = False
    b_vis_solver_name = True
    b_vis_fric_model = False
    b_vis_alpha = False

    # do global plot of states for entire log?
    b_global_plot = False
    glob_lim = np.inf

    # plot immediately or only solver data replay?
    b_immediate_plot_update = True

    # show plot of solver runtimes?
    b_calc_time_plot = False

    # save plots as tikz files?
    b_save_tikz = False

    # visulaization options
    vis_options = {'b_movie': b_movie,
                   'b_run_OSQP': b_run_OSQP,
                   'b_calc_qpOASES': b_calc_qpOASES,
                   'b_con_power': b_con_power,
                   'b_idx': b_idx,
                   'b_vis_model_name': b_vis_model_name,
                   # 'b_plot_course': b_plot_course,
                   'b_vis_solver_name': b_vis_solver_name,
                   'b_vis_fric_model': b_vis_fric_model,
                   'b_vis_alpha': b_vis_alpha,
                   'b_global_plot': b_global_plot,
                   'glob_lim': glob_lim,
                   'b_immediate_plot_update': b_immediate_plot_update,
                   'b_calc_time_plot': b_calc_time_plot,
                   'b_save_tikz': b_save_tikz}

    # --- Define solver options for IPOPT as benchmark solution
    sol_options = {'solver1': {'Model': "FW",               # PM (Pointmass), KM (Kinematic Single Track Model),
                                                            # DM (Dynamic Single Track Model), FW (Double Track Model)
                               'Solver': "IPOPT",           # IPOPT, OSQP, qpOASES
                               'Friction': "Diamond",       # Circle, Diamond (only for PM and KM)
                               'VarFriction': True,         # Variable friction: True, False
                               'VarPower': False,           # Variable power: True, False
                               'Slack': True,               # Usage of slack variables: True, False
                               }
                   }

    # --- Transform ID of used velocity planner into 'PerfSQP' or 'EmergSQP'
    sid = csv_name.split('/')[-1].split('_')[1]
    if sid == 'perf':
        sid = 'PerfSQP'
    elif sid == 'emerg':
        sid = 'EmergSQP'
    else:
        print('Logs have been produced with illegal SID! Exiting.')
        sys.exit(1)

    # --- Number of velocity points
    # Get length of velocity array to determine parameter 'm' in velocity optimization
    row_lc = linecache.getline(csv_name, 1)
    row_lc = row_lc[:-1].rsplit(';')
    velocity_dummy = json.loads(row_lc[2])
    m = len(velocity_dummy)

    # --- Set up visualization object
    rL = VisVP_Logs(csv_name=csv_name,
                    csv_name_ltpl=csv_name_ltpl,
                    m=m,
                    sid=sid,
                    log_lines=log_lines,
                    vis_options=vis_options,
                    params_path=params_path,
                    input_path=input_path,
                    sol_options=sol_options)

    # --- Start GUI
    rL.vis_log(int(0))
