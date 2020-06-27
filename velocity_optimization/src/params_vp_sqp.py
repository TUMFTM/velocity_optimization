import numpy as np
import configparser


def params_vp_sqp(m: int,
                  sid: str,
                  params_path: str) -> tuple:

    """
    Python version: 3.5
    Created by: Thomas Herrmann (thomas.herrmann@tum.de)
    Created on: 01.12.2019

    Documentation: Creates some random initialization values for the velocity planner SQP.

    Inputs:
    m: number of velocity points
    sid: ID of optimizer object 'PerfSQP' or 'EmergSQP'

    Outputs:
    sqp_stgs: SQP settings
    v_ini: initial velocity constraint [m/s]
    v_max: max. velocity (objective function) [m/s]
    v_end: end velocity in optimization horizon [m/s]
    x0_v: initial velocity guess [m/s]
    x0_s_t: initial slack guess [-]
    F_ini: initial force constraint [kN]
    kappa: curvature profile for path [rad/m]
    delta_s: discretization step length [m]
    P_max: max. allowed power [kW]
    ax_max: max. longitudinal acceleration [m/s^2]
    ay_max: max. lateral acceleration [m/s^2]
    err: SQP iteration RMSE [-]
    err_inf: SQP infinity error [-]
    """

    if m < 5:
        print("Optimization horizon too short! Increase m!")
        exit(1)

    # load config files
    sqp_config = configparser.ConfigParser()
    if not sqp_config.read(params_path + 'sqp_config.ini'):
        raise ValueError('Specified config file does not exist or is empty!')

    # --- SQP settings
    if sid == 'PerfSQP':
        id_str = 'SOLVER_PERFORMANCE'
    else:
        id_str = 'SOLVER_EMERGENCY'

    sqp_stgs = {'m': m,
                'slack_every_v': sqp_config.getint('SOLVER_GENERAL', 'slack_every_v'),
                'b_online_mode': sqp_config.getint('SOLVER_GENERAL', 'b_online_mode'),
                'b_var_power': sqp_config.getint(id_str, 'b_var_power'),
                'b_var_friction': sqp_config.getint(id_str, 'b_var_friction'),
                'b_vis_mode': sqp_config.getint('SOLVER_GENERAL', 'b_vis_mode'),
                'b_debug_mode': sqp_config.getint('SOLVER_GENERAL', 'b_debug_mode'),
                'b_print_n_qp': sqp_config.getint('SOLVER_GENERAL', 'b_print_n_qp'),
                'b_print_n_sqp': sqp_config.getint('SOLVER_GENERAL', 'b_print_n_sqp'),
                'b_print_QP_runtime': sqp_config.getint('SOLVER_GENERAL', 'b_print_QP_runtime'),
                'b_print_SQP_runtime': sqp_config.getint('SOLVER_GENERAL', 'b_print_SQP_runtime'),
                'b_solver_stat': sqp_config.getint('SOLVER_GENERAL', 'b_solver_stat'),
                'b_print_sqp_err': sqp_config.getint('SOLVER_GENERAL', 'b_print_sqp_err'),
                'b_print_sqp_alpha': sqp_config.getint('SOLVER_GENERAL', 'b_print_sqp_alpha'),
                'b_trajectory_check': sqp_config.getint('SOLVER_GENERAL', 'b_trajectory_check'),
                'b_print_s_v_val': sqp_config.getint('SOLVER_GENERAL', 'b_print_s_v_val'),
                'b_print_J': sqp_config.getint(id_str, 'b_print_J'),
                'b_print_condition_number': sqp_config.getint('SOLVER_GENERAL', 'b_print_condition_number'),
                'n_sqp_max': sqp_config.getint('SOLVER_GENERAL', 'n_sqp_max'),
                't_sqp_max': sqp_config.getfloat(id_str, 't_sqp_max'),
                'b_print_sm': sqp_config.getint('SOLVER_GENERAL', 'b_print_sm'),
                'b_sparse_matrix_fill': sqp_config.getint('SOLVER_GENERAL', 'b_sparse_matrix_fill')
                }

    # --- SQP termination criterion
    if sid == 'PerfSQP':
        err = 0.01 * sqp_stgs['m']
        err_inf = 0.01 * sqp_stgs['m']
    else:
        err = 0.01 * sqp_stgs['m'] * 3
        err_inf = 0.01 * sqp_stgs['m'] * 3

    # --- SQP initialization values
    # Initial velocity parameter [m/s]
    v_ini = 10
    # max. velocity in objective function [m/s]
    v_max = 70 * np.ones((sqp_stgs['m'], ))
    # end velocity in optimization horizon (hard) [m/s]
    v_end = 6.5
    # Force from previous optimization horizon acting on first acceleration (hard constraint) [kN]
    F_ini = 0.5

    # Initial velocity guess [m/s]
    x0_v = np.array([2, 3, 4])
    x0_v = np.append(x0_v, 5 * np.ones((sqp_stgs['m'] - 4), ))
    x0_v = np.insert(x0_v, 0, v_ini)

    # Initial guess slack variables tire [-]
    x0_s_t = np.zeros((int(np.ceil(sqp_stgs['m'] / sqp_stgs['slack_every_v'])), ))

    # Curvature profile of given path [rad/m]
    kappa = 0.0001 * np.ones((sqp_stgs['m'], ))
    # Discretization step length of given path [m]
    delta_s = 2.21 * np.ones((sqp_stgs['m'] - 1, ))

    # max. allowed power [kW]
    if sqp_stgs['b_var_power']:
        P_max = 270 * np.ones((sqp_stgs['m'] - 1, ))
    else:
        P_max = None

    # max. allowed long./lat. accelerations
    if sqp_stgs['b_var_friction']:
        ax_max = 12.5 * np.ones((sqp_stgs['m'], ))
        ay_max = 12.25 * np.ones((sqp_stgs['m'], ))
    else:
        ax_max = None
        ay_max = None

    return sqp_stgs, v_ini, v_max, v_end, x0_v, x0_s_t, F_ini, kappa, delta_s, P_max, ax_max, ay_max, err, err_inf
