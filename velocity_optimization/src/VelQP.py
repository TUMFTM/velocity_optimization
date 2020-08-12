try:
    from mpmath import mp
except ImportError:
    print("No module 'mpmath' found! Not necessary on car but for development.")
import numpy as np
import osqp as osqp
import scipy.sparse as sparse
import os
import datetime
import copy
import logging
import configparser
import json

from velocity_optimization.src.params_vp_sqp import params_vp_sqp
from velocity_optimization.src.VarPower import VarPowerLimits
from velocity_optimization.src.get_sparsity import calc_sparsity


class VelQP:

    __slots__ = ('sqp_stgs',
                 'm',
                 'params_path',
                 'slack_every_v',
                 'n',
                 'ev_vel_w',
                 'b_ini_done',
                 'sid',
                 'Am',
                 'lo',
                 'up',
                 'sym_sc_',
                 'ones_vec_red',
                 'E_mat_red',
                 'E_mat_redd',
                 'EP_mat_red',
                 'T_mat_inv',
                 'm_inv',
                 'err',
                 'err_inf',
                 'x0',
                 'x0_s_t',
                 'J_jac',
                 'J_Hess',
                 'F_cst',
                 'F_cst_jac',
                 'F_ini_cst',
                 'F_ini_cst_jac',
                 'v_cst_end',
                 'v_cst_end_jac',
                 'P_cst',
                 'P_cst_jac',
                 'Tre_cst1',
                 'Tre_cst1',
                 'Tre_cst1_jac',
                 'Tre_cst2',
                 'Tre_cst2_jac',
                 'Tre_cst3',
                 'Tre_cst3_jac',
                 'Tre_cst4',
                 'Tre_cst4_jac',
                 'dF_cst',
                 'dF_cst_jac',
                 'sparsity_pat',
                 'Am_csc',
                 'vpl',
                 'sol_osqp',
                 'logger_perf',
                 'logger_emerg')

    def __init__(self,
                 m: int,
                 sid: str,
                 params_path: str,
                 input_path: str,
                 logging_path: str = None,
                 ci: bool = False):
        """Class to construct QP-optimizer by a manually designed vector matrix notation.

        :param m: number of velocity points
        :param sid: optimized ID 'PerfSQP' or 'EmergSQP'
        :param params_path: absolute path to folder containing config file .ini
        :param input_path: absolute path to folder containing variable vehicle and track information
        :param ci: switch to construct an object of this class used within the CI/CD jobs

        :Authors:
            Thomas Herrmann <thomas.herrmann@tum.de>

        :Created on:
            01.11.2019
        """

        # --- Retrieve SQP settings from parameter-file
        self.sqp_stgs = params_vp_sqp(m=m,
                                      sid=sid,
                                      params_path=params_path)[0]

        # Store params_path
        self.params_path = params_path
        # number of velocity optimization variables
        self.m = m
        # counter for how many velocity variables one slack is valid
        self.slack_every_v = self.sqp_stgs['slack_every_v']
        # number of slack variables
        self.n = int(np.ceil(m / self.slack_every_v))
        # weight on velocity slacks
        self.ev_vel_w = None
        # determines whether A, P, u, l, q-matrices have been filled once
        self.b_ini_done = False
        # ID of optimizer object
        self.sid = sid

        ################################################################################################################
        # --- Pre-allocation of QP matrices
        ################################################################################################################
        # --- Matrix setup for Emergency SQP and Performance SQP
        if self.sid == 'EmergSQP':

            # --- no F_ini-constraint in emergency profile
            self.Am =\
                np.empty((1 * 1 + m * 0 + (m - 1) * 6 + (m - 2) * 1 + self.n * 1, m - 1 + self.n), dtype=np.float64)
            self.lo = np.empty((1 * 1 + m * 0 + (m - 1) * 6 + (m - 2) * 1 + self.n * 1, 1), dtype=np.float64)
            self.up = np.empty((1 * 1 + m * 0 + (m - 1) * 6 + (m - 2) * 1 + self.n * 1, 1), dtype=np.float64)

        else:

            self.Am =\
                np.empty((1 * 2 + m * 0 + (m - 1) * 5 + (m - 2) * 2 + self.n * 1, m - 1 + self.n), dtype=np.float64)
            self.lo = np.empty((1 * 2 + m * 0 + (m - 1) * 5 + (m - 2) * 2 + self.n * 1, 1), dtype=np.float64)
            self.up = np.empty((1 * 2 + m * 0 + (m - 1) * 5 + (m - 2) * 2 + self.n * 1, 1), dtype=np.float64)

        print('*** --- ' + sid + ' solver initialized --- ***')

        ################################################################################################################
        # --- Initialize Logging
        ################################################################################################################

        if logging_path is not None:
            # create logger for Performance SQP
            self.logger_perf = logging.getLogger('sqp_logger_perf')
            self.logger_perf.setLevel(logging.DEBUG)

            os.makedirs(logging_path, exist_ok=True)

            fh_perf = logging.FileHandler(logging_path + '/sqp_perf_'
                                          + datetime.datetime.now().strftime("%Y_%m_%d") + '_'
                                          + datetime.datetime.now().strftime("%H_%M")
                                          + '.log')
            fh_perf.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(message)s')
            fh_perf.setFormatter(formatter)
            self.logger_perf.addHandler(fh_perf)

            # create logger for Emergency SQP
            self.logger_emerg = logging.getLogger('sqp_logger_emerg')
            self.logger_emerg.setLevel(logging.DEBUG)
            fh_emerg = logging.FileHandler(logging_path + '/sqp_emerg_'
                                           + datetime.datetime.now().strftime("%Y_%m_%d") + '_'
                                           + datetime.datetime.now().strftime("%H_%M")
                                           + '.log')
            fh_emerg.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(message)s')
            fh_emerg.setFormatter(formatter)
            self.logger_emerg.addHandler(fh_emerg)

        ################################################################################################################
        # --- Assign numeric values to hard-coded parameters in QP
        ################################################################################################################

        # --- Performance SQP settings
        # load SQP config files
        sqp_config = configparser.ConfigParser()
        # load QP config sparsity patterns
        sqp_sparsity = configparser.ConfigParser()

        if not sqp_config.read(self.params_path + 'sqp_config.ini'):
            raise ValueError('Specified SQP config file does not exist or is empty!')

        # check whether we are running in CI/CD job
        if ci:
            calc_sparsity(params_path=params_path,
                          logging_path=logging_path,
                          m_perf=m,
                          m_emerg=m)

            src = logging_path + '/sparsity/.'
            dst = params_path

            os.system("cp -r -n " + src + " " + dst)

        if not sqp_sparsity.read(self.params_path + 'sqp_sparsity_' + sid + str(m) + '.ini') and \
                self.sqp_stgs['b_sparse_matrix_fill']:
            raise ValueError('Specified SQP sparsity file does not exist or is empty!')

        # --- Performance SQP settings
        if self.sid == 'PerfSQP':
            sym_sc_ = {'m_t_': sqp_config.getfloat('VEHICLE', 'mass_t'),
                       'Pmax_kW_': sqp_config.getfloat('VEHICLE', 'P_max_kW'),
                       'Fmax_kN_': sqp_config.getfloat('VEHICLE', 'F_max_kN'),
                       'Fmin_kN_': sqp_config.getfloat('VEHICLE', 'F_min_kN'),
                       'axmax_mps2_': sqp_config.getfloat('VEHICLE', 'ax_max_mps2'),
                       'aymax_mps2_': sqp_config.getfloat('VEHICLE', 'ay_max_mps2'),
                       'dF_kN_pos_': sqp_config.getfloat('SOLVER_GENERAL', 'dF_kN_pos'),
                       'dF_kN_neg_': sqp_config.getfloat('SOLVER_GENERAL', 'dF_kN_neg'),
                       'Fini_tol_': sqp_config.getfloat('SOLVER_PERFORMANCE', 'F_ini_tol'),
                       'c_res_': sqp_config.getfloat('VEHICLE', 'c_res'),
                       'vmin_mps_': sqp_config.getfloat('SOLVER_GENERAL', 'v_min_mps'),
                       's_v_t_lim_': sqp_config.getfloat('SOLVER_PERFORMANCE', 'slack_var_tire_lim'),
                       's_v_t_unit_': sqp_config.getfloat('SOLVER_GENERAL', 'slack_var_tire_unit'),
                       'dvdv_w_': sqp_config.getfloat('SOLVER_PERFORMANCE', 'penalty_jerk'),
                       'tre_cst_w_': sqp_config.getfloat('SOLVER_PERFORMANCE', 'w_tre_constraint'),
                       's_tre_w_lin_': sqp_config.getfloat('SOLVER_PERFORMANCE', 'penalty_slack_tire_lin'),
                       's_tre_w_quad_': sqp_config.getfloat('SOLVER_PERFORMANCE', 'penalty_slack_tire_quad')
                       }
            if self.sqp_stgs['b_sparse_matrix_fill']:
                self.sparsity_pat = {'F_ini_r': np.array(json.loads(sqp_sparsity.get('symqp.F_sym_ini_cst_jac_:_1:', 'r'))),  # noqa: E501
                                     'F_ini_c': np.array(json.loads(sqp_sparsity.get('symqp.F_sym_ini_cst_jac_:_1:', 'c'))),  # noqa: E501
                                     'F_r': np.array(json.loads(sqp_sparsity.get('symqp.F_sym_cst_jac_1:_1:', 'r'))),
                                     'F_c': np.array(json.loads(sqp_sparsity.get('symqp.F_sym_cst_jac_1:_1:', 'c'))),
                                     'P_r': np.array(json.loads(sqp_sparsity.get('symqp.P_sym_cst_jac_:_1:', 'r'))),
                                     'P_c': np.array(json.loads(sqp_sparsity.get('symqp.P_sym_cst_jac_:_1:', 'c'))),
                                     'Tre_r': np.array(json.loads(sqp_sparsity.get('symqp.Tre_sym_cst1_jac_:_1:', 'r'))),  # noqa: E501
                                     'Tre_c': np.array(json.loads(sqp_sparsity.get('symqp.Tre_sym_cst1_jac_:_1:', 'c')))
                                     }

        # --- Emergency SQP settings
        else:
            sym_sc_ = {'m_t_': sqp_config.getfloat('VEHICLE', 'mass_t'),
                       'Pmax_kW_': sqp_config.getfloat('VEHICLE', 'P_max_kW'),
                       'Fmax_kN_': sqp_config.getfloat('VEHICLE', 'F_max_kN'),
                       'Fmin_kN_': sqp_config.getfloat('VEHICLE', 'F_min_kN'),
                       'axmax_mps2_': sqp_config.getfloat('VEHICLE', 'ax_max_mps2'),
                       'aymax_mps2_': sqp_config.getfloat('VEHICLE', 'ay_max_mps2'),
                       'dF_kN_pos_': sqp_config.getfloat('SOLVER_GENERAL', 'dF_kN_pos'),
                       'dF_kN_neg_': sqp_config.getfloat('SOLVER_GENERAL', 'dF_kN_neg'),
                       'Fini_tol_': sqp_config.getfloat('SOLVER_EMERGENCY', 'F_ini_tol'),
                       'c_res_': sqp_config.getfloat('VEHICLE', 'c_res'),
                       'vmin_mps_': sqp_config.getfloat('SOLVER_GENERAL', 'v_min_mps'),
                       's_v_t_lim_': sqp_config.getfloat('SOLVER_EMERGENCY', 'slack_var_tire_lim'),
                       's_v_t_unit_': sqp_config.getfloat('SOLVER_GENERAL', 'slack_var_tire_unit'),
                       'dvdv_w_': sqp_config.getfloat('SOLVER_EMERGENCY', 'penalty_jerk'),
                       'tre_cst_w_': sqp_config.getfloat('SOLVER_EMERGENCY', 'w_tre_constraint'),
                       's_tre_w_lin_': sqp_config.getfloat('SOLVER_EMERGENCY', 'penalty_slack_tire_lin'),
                       's_tre_w_quad_': sqp_config.getfloat('SOLVER_EMERGENCY', 'penalty_slack_tire_quad')
                       }
            if self.sqp_stgs['b_sparse_matrix_fill']:
                self.sparsity_pat = {'F_r': np.array(json.loads(sqp_sparsity.get('symqp.F_sym_cst_jac_:_1:', 'r'))),
                                     'F_c': np.array(json.loads(sqp_sparsity.get('symqp.F_sym_cst_jac_:_1:', 'c'))),
                                     'P_r': np.array(json.loads(sqp_sparsity.get('symqp.P_sym_cst_jac_:_1:', 'r'))),
                                     'P_c': np.array(json.loads(sqp_sparsity.get('symqp.P_sym_cst_jac_:_1:', 'c'))),
                                     'Tre_r': np.array(json.loads(sqp_sparsity.get('symqp.Tre_sym_cst1_jac_:_1:', 'r'))),  # noqa: E501
                                     'Tre_c': np.array(json.loads(sqp_sparsity.get('symqp.Tre_sym_cst1_jac_:_1:', 'c')))
                                     }

        # Assign settings to SQP object
        self.sym_sc_ = sym_sc_

        ################################################################################################################
        # --- Pre-define numeric matrices
        ################################################################################################################

        # Vector with elements 1, ..., 1: m-1 x 1
        self.ones_vec_red = np.ones((m - 1, 1), dtype=np.float64)
        # unity matrix last row removed: m-1 x m+n
        self.E_mat_red = np.eye(m - 1, m + self.n, dtype=np.float64)
        # unity matrix last 2 rows removed: m-2 x m+n
        self.E_mat_redd = np.eye(m - 2, m + self.n, dtype=np.float64)
        # unity matrix last row removed: m-1 x m+n; filled with power constraint
        self.EP_mat_red = np.eye(m - 1, m + self.n, dtype=np.float64)
        # Matrix to store inverse elements of delta t for every discretization step
        self.T_mat_inv = np.zeros((m - 1, m + self.n), dtype=np.float64)

        # --- Contribution of derivative of tire slack variables to tire jacobian
        for j in range(self.n):
            self.T_mat_inv[j * self.slack_every_v:self.slack_every_v + j * self.slack_every_v, - self.n + j] =\
                - 1 * self.sym_sc_['s_v_t_unit_']

        # Inverse of vehicle mass [1/t = 1/1000kg]
        self.m_inv = 1 / sym_sc_['m_t_']

        # Initialization of attributes
        self.err = None
        self.err_inf = None
        self.x0 = None
        self.x0_s_t = None

        self.J_jac = None
        self.J_Hess = None
        self.F_cst = None
        self.F_cst_jac = None
        self.F_ini_cst = None
        self.F_ini_cst_jac = None
        self.v_cst_end = None
        self.v_cst_end_jac = None
        self.P_cst = None
        self.P_cst_jac = None
        self.Tre_cst1 = None
        self.Tre_cst1 = None
        self.Tre_cst1_jac = None
        self.Tre_cst2 = None
        self.Tre_cst2_jac = None
        self.Tre_cst3 = None
        self.Tre_cst3_jac = None
        self.Tre_cst4 = None
        self.Tre_cst4_jac = None
        self.dF_cst = None
        self.dF_cst_jac = None

        self.Am_csc = None

        ################################################################################################################
        # --- Get variable power handler class
        ################################################################################################################
        self.vpl = VarPowerLimits(input_path=input_path)

        ################################################################################################################
        # --- Construct QP solver object (OSQP)
        ################################################################################################################
        self.sol_osqp = osqp.OSQP()

        ################################################################################################################
        # --- Initialize QP solver object (OSQP)
        ################################################################################################################
        self.osqp_init()

    def osqp_init(self):
        """Initializes the QP solver OSQP and does the solver settings.\n
        OSQP solves the problem

        .. math::
            1/2~x^T P x + q^T x \n
            \mathrm{s.t.} \quad l \leq Ax \leq u # noqa: W605

        :Authors:
            Thomas Herrmann <thomas.herrmann@tum.de>

        :Created on:
            01.11.2019
        """

        # --- Retrieve some initialization values for solver (depending on variable power and variable friction pot.)
        # --- Check units of retrieved parameters in params_vp_sqp() documentation
        sqp_stgs, \
            v_ini, \
            v_max, \
            v_end, \
            x0_v, \
            x0_s_t, \
            F_ini, \
            kappa, \
            delta_s, \
            P_max, \
            ax_max, \
            ay_max, \
            err, \
            err_inf = params_vp_sqp(m=self.m,
                                    sid=self.sid,
                                    params_path=self.params_path)

        # SQP termination criterion: RMSE
        self.err = err
        # SQP termination criterion: infinity error
        self.err_inf = err_inf

        P, q, A, lo, up = self.get_osqp_mat(x0_v=x0_v,
                                            x0_s_t=x0_s_t,
                                            v_ini=v_ini,
                                            v_max=v_max,
                                            v_end=v_end,
                                            F_ini=F_ini,
                                            kappa=kappa,
                                            delta_s=delta_s,
                                            P_max=P_max,
                                            ax_max=ax_max,
                                            ay_max=ay_max)

        if sqp_stgs['b_online_mode']:

            if self.sid == 'EmergSQP':
                sol_osqp_stgs = {'verbose': False,
                                 'eps_abs': 3e-2,
                                 'eps_rel': 1e-2,
                                 'polish': False,
                                 'max_iter': 1000,
                                 'scaled_termination': True
                                 }
            else:
                sol_osqp_stgs = {'verbose': False,
                                 'eps_abs': 1e-2,
                                 'eps_rel': 1e-2,
                                 'polish': False,
                                 'max_iter': 1000,
                                 'scaled_termination': True
                                 }
        else:
            sol_osqp_stgs = {'verbose': True,
                             'eps_abs': 1e-2,
                             'eps_rel': 1e-2,
                             'polish': False,
                             'max_iter': 2000,
                             'scaled_termination': True
                             }

        self.sol_osqp.setup(P=P, q=q, A=A, l=lo, u=up, **sol_osqp_stgs)  # noqa: E741

    def osqp_solve(self) -> tuple:

        """Solves the constructed QP and returns the solution as well as OSQP status information.

        :return: sol.x: optimization variable values\n
            sol.info.iter: iteration number of OSQP solver\n
            sol.info.status_val: status of OSQP solver

        :Authors:
            Thomas Herrmann <thomas.herrmann@tum.de>

        :Created on:
            01.11.2019
        """

        # --- Solve constructed QP
        sol = self.sol_osqp.solve()

        # Print OSQP solver runtime
        if self.sqp_stgs['b_print_QP_runtime']:
            print(self.sid + " | QP time [ms]: ", sol.info.run_time * 1000)
        if self.sqp_stgs['b_print_n_qp']:
            print(self.sid + " | QP iter.:", sol.info.iter)

        # Print OSQP solver status for inaccuracy or max. iterations reached
        if self.sqp_stgs['b_solver_stat'] == 1:
            if sol.info.status_val == 2:
                print(self.sid + " | QP solved inaccurately!")
                # print(sol.x)
            elif sol.info.status_val == -2:
                print(self.sid + " | max. QP iterations reached!")
            elif sol.info.status_val == -3:
                print(self.sid + " | primal infeasible!")

        return sol.x, sol.info.iter, sol.info.status_val

    def osqp_update_online(self,
                           x0_v: np.ndarray,
                           x0_s_t: np.ndarray,
                           v_ini: float,
                           v_max: np.ndarray,
                           v_end: float,
                           F_ini: float,
                           kappa: np.ndarray,
                           delta_s: np.ndarray,
                           P_max: np.ndarray = None,
                           ax_max: np.ndarray = None,
                           ay_max: np.ndarray = None,
                           vmax_cstr: np.ndarray = None):

        """Updates the matrices in the constructed QP.

        :param x0_v: initial guess velocity [m/s]
        :param x0_s_t: initial guess slack variables tire [-]
        :param v_ini: initial hard constrained velocity [m/s]
        :param v_max: max. allowed velocity (in objective function) [m/s]
        :param v_end: hard constrained max. allowed value of end velocity in optimization horizon [m/s]
        :param F_ini: hard constrained initial force [kN]
        :param kappa: curvature profile of given path [rad/m]
        :param delta_s: discretization step length of given path [m]
        :param P_max: max. allowed power [kW]
        :param ax_max: max. allowed longitudinal acceleration [m/s^2]
        :param ay_max: max. allowed lateral accelereation [m/s]
        :param vmax_cstr: max. allowed spatially dependend velocity (hard constraint) [m/s^2]; can be used for
            e.g. adaptive cruise control (following)

        :Authors:
            Thomas Herrmann <thomas.herrmann@tum.de>

        :Created on:
            01.11.2019
        """

        if (self.sqp_stgs['b_var_friction'] and ax_max is None) or (self.sqp_stgs['b_var_friction'] and ay_max is None):
            print('Error! Variable friction specified but friction arrays are empty in ' + self.sid + '!')

        # --- retrieve OSQP solver matrices
        P, q, A, lo, up = self.get_osqp_mat(x0_v, x0_s_t, v_ini, v_max, v_end, F_ini, kappa, delta_s,
                                            P_max, ax_max, ay_max, vmax_cstr)

        # --- Update OSQP-problem matrices (Px, Ax is data from sparse matrices)
        self.sol_osqp.update(q=q, Ax=A.data, l=lo, u=up)  # noqa: E741

    def get_osqp_mat(self,
                     x0_v: np.ndarray,
                     x0_s_t: np.ndarray,
                     v_ini: float,
                     v_max: np.ndarray,
                     v_end: float,
                     F_ini: float,
                     kappa: np.ndarray,
                     delta_s: np.ndarray,
                     P_max: np.ndarray = None,
                     ax_max: np.ndarray = None,
                     ay_max: np.ndarray = None,
                     v_max_cstr: np.ndarray = None) -> tuple:

        """Constructs necessary QP matrices from the linearized versions of the physically necessary
        equations for the velocity optimization of a vehicle.

        :param x0_v: initial guess velocity [m/s]
        :param x0_s_t: initial guess slack variables tire [-]
        :param v_ini: initial hard constrained velocity [m/s]; currently unused here as we remove the first velocity
            point from the optimization problem and and set it manually afterwards
        :param v_max: max. allowed velocity (in objective function) [m/s]
        :param v_end: hard constrained max. allowed value of end velocity in optimization horizon [m/s]
        :param F_ini: hard constrained initial force [kN]
        :param kappa: curvature profile of given path [rad/m]
        :param delta_s: discretization step length of given path [m]
        :param P_max: max. allowed power [kW]
        :param ax_max: max. allowed longitudinal acceleration [m/s^2]
        :param ay_max: max. allowed lateral accelereation [m/s^2]
        :param vmax_cstr: max. allowed spatially dependend velocity (hard constraint) [m/s]; can be used for
            e.g. adaptive cruise control (following)

        :return: P: Sparse version of Hessian matrix of nonlinear objective function J\n
            q: Jacobian of nonlinear objective function J\n
            Am_csc: Sparse version of jacobian of nonlinear constraints\n
            lo: lower boundaries of linearized constraints\n
            up: upper boundaries of linearized constraints

        :Authors:
            Thomas Herrmann <thomas.herrmann@tum.de>

        :Created on:
            01.11.2019
        """

        m = self.m
        n = self.n

        ################################################################################################################
        # Pre-calculations: need to be done online but only once within this loop
        ################################################################################################################

        # Inverse of discretization step lengths [1/m]
        delta_s_inv = 1 / delta_s
        # Inverse of discretization step lengths * 1/2 [1/m]
        delta_2s_inv = 0.5 * delta_s_inv
        # Initial velocity guess squared [m^2/s^2]
        x0_squ = np.square(x0_v[0:m - 1])
        # Initial velocity guess cubed [m^3/s^3]
        x0_cub = x0_v[0:m - 1] ** 3
        # Initial velocity guess squared reduced by first entry [m^2/s^2]
        x0_squ_wh = np.square(x0_v[1:m])
        # Initial velocity guess reduced by last entry [m/s]
        x0_red = x0_v[0:m - 1]
        # Initial velocity guess reduced by first entry [m/s]
        x0_wh = x0_v[1:m]
        # curvature profile reduced by last entry [rad/m]
        kappa = kappa[0:m - 1]

        # Inverse of friction limits [s^2/m] (constant or spatially variable)
        if ax_max is not None:
            ax_max_inv = np.divide(np.ones((1, m - 1), dtype=np.float64), ax_max[0:-1])
            ay_max_inv = np.divide(np.ones((1, m - 1), dtype=np.float64), ay_max[0:-1])
        else:
            ax_max_inv = 1 / self.sym_sc_['axmax_mps2_']
            ay_max_inv = 1 / self.sym_sc_['aymax_mps2_']

        E_mat_red = self.E_mat_red
        E_mat_redd = self.E_mat_redd
        EP_mat_red = self.EP_mat_red
        T_mat_inv = self.T_mat_inv

        ################################################################################################################
        # Calculate matrices for OSQP solver
        ################################################################################################################

        self.x0 = x0_v
        self.x0_s_t = x0_s_t

        # Inverse of delta t for every discretization step length [1/s]
        t_inv = x0_red * delta_s_inv

        # Inverse of delta t for every discretization step length [1/s] with shifted velocity entries: v1/d0, v2/d1, ...
        t_inv_wh = x0_wh * delta_s_inv

        # --- Construction of jacobian of objective function J
        # fJ_jac(x0_v, v_max)
        self.J_jac = np.zeros(m, dtype=np.float64)

        # first 2 lines
        self.J_jac[0] = self.sym_sc_['dvdv_w_'] * \
            (2 * x0_v[0] - 4 * x0_v[1] + 2 * x0_v[2]) + 2 * (x0_v[0] - v_max[0])

        self.J_jac[1] = self.sym_sc_['dvdv_w_'] * \
            ((2 * x0_v[1] - 4 * x0_v[2] + 2 * x0_v[3]) + (- 4 * x0_v[0] + 8 * x0_v[1] - 4 * x0_v[2])) + \
            2 * (x0_v[1] - v_max[1])

        # mid part
        self.J_jac[2:-2] = self.sym_sc_['dvdv_w_'] * \
            (2 * x0_v[0:-4] - 8 * x0_v[1:-3] + 12 * x0_v[2:-2] - 8 * x0_v[3:-1] + 2 * x0_v[4:]) + \
            2 * (x0_v[2:-2] - v_max[2:-2])
        # last 2 lines
        self.J_jac[-2] = self.sym_sc_['dvdv_w_'] * \
            ((2 * x0_v[-4] - 4 * x0_v[-3] + 2 * x0_v[-2]) + (-4 * x0_v[-3] + 8 * x0_v[-2] - 4 * x0_v[-1])) + \
            2 * (x0_v[-2] - v_max[-2])

        self.J_jac[-1] = self.sym_sc_['dvdv_w_'] * \
            (2 * x0_v[-3] - 4 * x0_v[-2] + 2 * x0_v[-1]) + 2 * (x0_v[-1] - v_max[-1])

        # Slack-contribution
        self.J_jac =\
            np.append(self.J_jac, 2 * self.sym_sc_['s_tre_w_quad_'] * self.sym_sc_['s_v_t_unit_'] ** 2 * x0_s_t
                      + self.sym_sc_['s_tre_w_lin_'] * self.sym_sc_['s_v_t_unit_'])

        # --- Do not convert Hessian if not necessary (e.g., if constant)
        if self.J_Hess is None:

            self.J_Hess = np.eye(m + n, dtype=np.float64)

            h_diag = 12 * self.sym_sc_['dvdv_w_'] * np.ones(m, dtype=np.float64)
            h_diag[0] = 2 * self.sym_sc_['dvdv_w_']
            h_diag[1] = 10 * self.sym_sc_['dvdv_w_']
            h_diag[-2] = 10 * self.sym_sc_['dvdv_w_']
            h_diag[-1] = 2 * self.sym_sc_['dvdv_w_']
            np.fill_diagonal(self.J_Hess[:m, :], h_diag + 2)

            h_diag_k1 = -8 * self.sym_sc_['dvdv_w_'] * np.ones(m - 1, dtype=np.float64)
            h_diag_k1[0] = -4 * self.sym_sc_['dvdv_w_']
            h_diag_k1[-1] = -4 * self.sym_sc_['dvdv_w_']
            np.fill_diagonal(self.J_Hess[:m - 1, 1:], h_diag_k1)
            np.fill_diagonal(self.J_Hess[1:m], h_diag_k1)

            h_diag_k2 = 2 * self.sym_sc_['dvdv_w_'] * np.ones(m - 2, dtype=np.float64)
            np.fill_diagonal(self.J_Hess[:m - 2, 2:], h_diag_k2)
            np.fill_diagonal(self.J_Hess[2:m], h_diag_k2)

            s_diag = 2 * self.sym_sc_['s_tre_w_quad_'] * self.sym_sc_['s_v_t_unit_'] ** 2 * np.ones((n, ))
            np.fill_diagonal(self.J_Hess[m:, m:], s_diag)

            # Print condition number of Hessian, useful for debugging
            if self.sqp_stgs['b_print_condition_number']:
                S = mp.svd_r(mp.matrix(self.J_Hess), compute_uv=False)
                print(self.sid)
                print("Singular values of Hessian in " + self.sid, S)
                cond_number = np.max(S) / np.min(S)
                print("Condition-number of Hessian:", cond_number)

        else:
            pass

        # linearized force box constraint [kN], fF_cst(x0_v, delta_s)
        self.F_cst = (- self.sym_sc_['Fmax_kN_'] * np.ones((1, m - 1))
                      + 0.001 * self.sym_sc_['c_res_'] * x0_squ
                      + self.sym_sc_['m_t_'] * (x0_squ_wh - x0_squ) * delta_2s_inv).T

        # jacobian
        np.fill_diagonal(E_mat_red[:, 1:], self.sym_sc_['m_t_'] * t_inv_wh)
        np.fill_diagonal(E_mat_red, 0.002 * self.sym_sc_['c_res_'] * x0_red - self.sym_sc_['m_t_'] * t_inv)
        self.F_cst_jac = E_mat_red

        # initial force constraint including small tolerance [kN]
        self.F_ini_cst = - F_ini - self.sym_sc_['Fini_tol_'] + \
            0.001 * self.sym_sc_['c_res_'] * x0_squ[0] + \
            self.sym_sc_['m_t_'] * (x0_squ_wh[0] - x0_squ[0]) * delta_2s_inv[0]

        # jacobian
        self.F_ini_cst_jac = np.zeros((1, m + n), dtype=np.float64)
        self.F_ini_cst_jac[0, 0] = 0.002 * self.sym_sc_['c_res_'] * x0_v[0] - self.sym_sc_['m_t_'] * t_inv[0]
        self.F_ini_cst_jac[0, 1] = self.sym_sc_['m_t_'] * t_inv_wh[0]

        # max. end velocity hard constraint [m/s]
        self.v_cst_end = x0_v[-1] - v_end

        # constant jacobian
        if not self.b_ini_done:
            self.v_cst_end_jac = np.zeros((1, m + self.n), dtype=np.float64)
            self.v_cst_end_jac[- 1, - 1 - n] = 1

        # power constraint [kW]
        if self.sqp_stgs['b_var_power']:
            self.P_cst = (- P_max
                          + (self.sym_sc_['m_t_'] * x0_red * (x0_squ_wh - x0_squ) * delta_2s_inv)
                          + self.sym_sc_['c_res_'] * x0_cub * 0.001).reshape((self.m - 1, 1))
        else:
            self.P_cst = (- self.sym_sc_['Pmax_kW_'] * np.ones((m - 1, 1)).T
                          + (self.sym_sc_['m_t_'] * x0_red * (x0_squ_wh - x0_squ) * delta_2s_inv)
                          + self.sym_sc_['c_res_'] * x0_cub * 0.001).T

        # jacobian
        pw1_ = 0.001 * self.sym_sc_['c_res_'] * x0_squ + \
            x0_red * (0.002 * self.sym_sc_['c_res_'] * x0_red - self.sym_sc_['m_t_'] * t_inv) + \
            self.sym_sc_['m_t_'] * (x0_squ_wh - x0_squ) * delta_2s_inv
        pw2_ = self.sym_sc_['m_t_'] * x0_red * x0_v[1:m] * delta_s_inv
        np.fill_diagonal(EP_mat_red, pw1_)
        np.fill_diagonal(EP_mat_red[:, 1:], pw2_)
        self.P_cst_jac = EP_mat_red

        # slack contribution values
        s_t_len = len(self.ones_vec_red)
        s_t_contrib = \
            np.reshape(np.repeat(self.sym_sc_['s_v_t_unit_'] * x0_s_t, self.slack_every_v)[0:s_t_len], (s_t_len, 1))

        tre_dv = (x0_squ_wh - x0_squ) * delta_2s_inv * ax_max_inv + \
            0.001 * self.sym_sc_['c_res_'] * x0_squ * ax_max_inv * self.m_inv
        tre_v2 = kappa * x0_squ * ay_max_inv
        # Tire diamond constraint [-]
        self.Tre_cst1 = (tre_dv + tre_v2 - self.ones_vec_red.T).T
        self.Tre_cst1 -= s_t_contrib

        # jacobian
        p1_ = 2 * kappa * x0_red * ay_max_inv
        p2_ = (0.002 * self.sym_sc_['c_res_'] * x0_red - self.sym_sc_['m_t_'] * t_inv) * \
            (ax_max_inv / self.sym_sc_['m_t_'])
        np.fill_diagonal(T_mat_inv[:, 1:], t_inv_wh * ax_max_inv)
        T_mat_inv_tre = T_mat_inv
        T_mat_inv_tre_neg = - T_mat_inv
        # keep the "-1" in these entries fixed (stemming from the slack variables)
        T_mat_inv_tre_neg[:, - self.n:] *= - 1

        np.fill_diagonal(T_mat_inv_tre, p1_ + p2_)
        self.Tre_cst1_jac = copy.deepcopy(T_mat_inv_tre)

        # Tire diamond constraint [-]
        self.Tre_cst2 = (- tre_dv + tre_v2 - self.ones_vec_red.T).T
        self.Tre_cst2 -= s_t_contrib

        # jacobian
        np.fill_diagonal(T_mat_inv_tre_neg, p1_ - p2_)
        self.Tre_cst2_jac = copy.deepcopy(T_mat_inv_tre_neg)

        # Tire diamond constraint [-]
        self.Tre_cst3 = (- tre_dv - tre_v2 - self.ones_vec_red.T).T
        self.Tre_cst3 -= s_t_contrib

        # jacobian
        np.fill_diagonal(T_mat_inv_tre_neg, - p1_ - p2_)
        self.Tre_cst3_jac = copy.deepcopy(T_mat_inv_tre_neg)

        # Tire diamond constraint [-]
        self.Tre_cst4 = (+ tre_dv - tre_v2 - self.ones_vec_red.T).T
        self.Tre_cst4 -= s_t_contrib

        # jacobian
        np.fill_diagonal(T_mat_inv_tre, - p1_ + p2_)
        self.Tre_cst4_jac = copy.deepcopy(T_mat_inv_tre)

        # delta Force constraint [kN]
        self.dF_cst = - self.sym_sc_['dF_kN_pos_'] * np.ones((m - 2, 1)) + self.F_cst[1:m] - self.F_cst[0:m - 2]

        # jacobian
        d1_ = self.sym_sc_['m_t_'] * t_inv[0:m - 2] - 0.002 * self.sym_sc_['c_res_'] * x0_red[0:m - 2]
        d2_ = \
            self.sym_sc_['m_t_'] * (- t_inv[1:m - 1] - t_inv_wh[0:m - 2]) + \
            0.002 * self.sym_sc_['c_res_'] * x0_wh[0:m - 2]
        d3_ = self.sym_sc_['m_t_'] * t_inv_wh[1:m]

        np.fill_diagonal(E_mat_redd, d1_)
        np.fill_diagonal(E_mat_redd[:, 1:], d2_)
        np.fill_diagonal(E_mat_redd[:, 2:], d3_)
        self.dF_cst_jac = E_mat_redd

        ################################################################################################################
        # P, sparse CSC
        ################################################################################################################

        if not self.b_ini_done:
            # Exclude first velocity point
            P = sparse.csc_matrix(self.J_Hess[1:, 1:])
        else:
            P = None

        ################################################################################################################
        # q
        ################################################################################################################

        # Exclude first velocity point
        q = self.J_jac.T[1:]

        ################################################################################################################
        # A, sparse CSC
        ################################################################################################################

        # --- Initialize sparse matrix A only once with following sparsity pattern
        if self.Am_csc is None or not self.sqp_stgs['b_sparse_matrix_fill']:

            ir_ = 0
            # Lower box on every velocity value to be > 0 (for numerical issues in emergency profile)
            if not self.b_ini_done:

                # Velocity floor constraint to avoid v < 0 on v__1, ..., v__m-1 (in total m-2 points)
                self.Am[ir_:ir_ + m - 1, :] = 0
                self.Am[ir_:ir_ + m - 2, 0:m - 2] = np.eye(m - 2, m - 2, dtype=np.float64)
                ir_ += m - 2

                # Slack ceil constraint to ensure slack < specified value and slack > 0
                self.Am[ir_:ir_ + n, :] = 0
                self.Am[ir_:ir_ + n, - n:] = self.sym_sc_['s_v_t_unit_'] * np.eye(n, n, dtype=np.float64)
                ir_ += n

                # end velocity constraint: v_m < v_end
                self.Am[ir_, :] = 0
                self.Am[ir_, 0:m - 1] = self.v_cst_end_jac[0, 1:m]
                ir_ += 1

            # --- Only move pointer ir_ to valid position
            else:
                ir_ += m - 2
                ir_ += n
                ir_ += 1

            # --- Don't fill Emergency SQP with F_ini constraint
            if self.sid != 'EmergSQP':

                self.Am[ir_:ir_ + 1, :] = 0.1 * self.F_ini_cst_jac[:, 1:]
                ir_ += 1

                # Fill PerfSQP with F-box-constraint apart from first force point
                self.Am[ir_:ir_ + m - 2, 0:m - 1 + n] = 0.1 * self.F_cst_jac[1:, 1:]
                ir_ += m - 2

            # --- Fill Emergency SQP with force box constraint
            else:

                self.Am[ir_:ir_ + m - 1, 0:m - 1 + n] = 0.1 * self.F_cst_jac[:, 1:]
                ir_ += m - 1

            # Power constraint
            self.Am[ir_:ir_ + m - 1, 0:m - 1 + n] = 0.01 * self.P_cst_jac[:, 1:]
            ir_ += m - 1

            # Tire constraint
            self.Am[ir_:ir_ + m - 1, 0:m - 1 + n] = self.sym_sc_['tre_cst_w_'] * self.Tre_cst1_jac[:, 1:]
            ir_ += m - 1

            # Tire constraint
            self.Am[ir_:ir_ + m - 1, 0:m - 1 + n] = self.sym_sc_['tre_cst_w_'] * self.Tre_cst2_jac[:, 1:]
            ir_ += m - 1

            # Tire constraint
            self.Am[ir_:ir_ + m - 1, 0:m - 1 + n] = self.sym_sc_['tre_cst_w_'] * self.Tre_cst3_jac[:, 1:]
            ir_ += m - 1

            # Tire constraint
            self.Am[ir_:ir_ + m - 1, 0:m - 1 + n] = self.sym_sc_['tre_cst_w_'] * self.Tre_cst4_jac[:, 1:]
            ir_ += m - 1

            # delta Force hard constraint (currently included in objective funciton)
            # self.Am[ir_:ir_ + m-2, 0:m-1] = self.dF_cst_jac[:,1:]  # A = np.append(A, self.dF_pos_cst_jac_, axis=0)
            # ir += m-2

        # --- If sparse version of matrix Am is available, fill sparsity pattern
        else:

            # --- Skip constant entries in Am_csc
            ir_ = 0
            ir_ += m - 2
            ir_ += n
            ir_ += 1

            # --- Don't fill Emergency SQP with F_ini constraint
            if self.sid != 'EmergSQP':

                self.Am_csc[ir_ + self.sparsity_pat['F_ini_r'], self.sparsity_pat['F_ini_c']] = \
                    0.1 * self.F_ini_cst_jac[:, 1:][self.sparsity_pat['F_ini_r'], self.sparsity_pat['F_ini_c']]
                ir_ += 1

                # Fill PerfSQP with F-box-constraint apart from first force point
                self.Am_csc[ir_ + self.sparsity_pat['F_r'], self.sparsity_pat['F_c']] = \
                    0.1 * self.F_cst_jac[1:, 1:][self.sparsity_pat['F_r'], self.sparsity_pat['F_c']]
                ir_ += m - 2

            else:
                # Force box constraint
                self.Am_csc[ir_ + self.sparsity_pat['F_r'], self.sparsity_pat['F_c']] = \
                    0.1 * self.F_cst_jac[:, 1:][self.sparsity_pat['F_r'], self.sparsity_pat['F_c']]
                ir_ += m - 1

            # Power constraint
            self.Am_csc[ir_ + self.sparsity_pat['P_r'], self.sparsity_pat['P_c']] = \
                0.01 * self.P_cst_jac[:, 1:][self.sparsity_pat['P_r'], self.sparsity_pat['P_c']]
            ir_ += m - 1

            # Tire constraint
            self.Am_csc[ir_ + self.sparsity_pat['Tre_r'], self.sparsity_pat['Tre_c']] = \
                self.sym_sc_['tre_cst_w_'] * \
                self.Tre_cst1_jac[:, 1:][self.sparsity_pat['Tre_r'], self.sparsity_pat['Tre_c']]
            ir_ += m - 1

            # Tire constraint
            self.Am_csc[ir_ + self.sparsity_pat['Tre_r'], self.sparsity_pat['Tre_c']] = \
                self.sym_sc_['tre_cst_w_'] * \
                self.Tre_cst2_jac[:, 1:][self.sparsity_pat['Tre_r'], self.sparsity_pat['Tre_c']]
            ir_ += m - 1

            # Tire constraint
            self.Am_csc[ir_ + self.sparsity_pat['Tre_r'], self.sparsity_pat['Tre_c']] = \
                self.sym_sc_['tre_cst_w_'] * \
                self.Tre_cst3_jac[:, 1:][self.sparsity_pat['Tre_r'], self.sparsity_pat['Tre_c']]
            ir_ += m - 1

            # Tire constraint
            self.Am_csc[ir_ + self.sparsity_pat['Tre_r'], self.sparsity_pat['Tre_c']] = \
                self.sym_sc_['tre_cst_w_'] * \
                self.Tre_cst4_jac[:, 1:][self.sparsity_pat['Tre_r'], self.sparsity_pat['Tre_c']]
            ir_ += m - 1

            # delta Force constraint
            # self.Am[ir_:ir_ + m-2, 0:m-1] = self.dF_cst_jac[:,1:]  # A = np.append(A, self.dF_pos_cst_jac_, axis=0)
            # ir += m-2

        if self.Am_csc is None or not self.sqp_stgs['b_sparse_matrix_fill']:
            # define A as sparse CSC
            self.Am_csc = sparse.csc_matrix(self.Am)

        ################################################################################################################
        # l, val_min - val_max - g(x0)
        ################################################################################################################
        ir_ = 0

        # Lower box on every velocity to avoid v < 0 on v__1, ..., v__m-1
        self.lo[ir_:ir_ + m - 2, 0] = - x0_wh[:-1]
        ir_ += m - 2

        # Floor '0' on tire slack variables
        self.lo[ir_:ir_ + n, 0] = - self.sym_sc_['s_v_t_unit_'] * x0_s_t
        ir_ += n

        # End velocity lower bound
        if not self.b_ini_done:
            self.lo[ir_, 0] = - np.inf
        ir_ += 1

        # --- Don't fill Emergency SQP with F_ini constraint
        if self.sid != 'EmergSQP':

            # F_ini lower bound
            self.lo[ir_, 0] = 0.1 * (- self.F_ini_cst - 2 * self.sym_sc_['Fini_tol_'])
            ir_ += 1

            # Force box lower bound
            self.lo[ir_:ir_ + m - 2, 0] = 0.1 * \
                ([self.sym_sc_['Fmin_kN_'] - self.sym_sc_['Fmax_kN_']] * (m - 2) - self.F_cst[1:, :].T)
            ir_ += m - 2

        else:
            # Force box lower bound
            self.lo[ir_:ir_ + m - 1, 0] = 0.1 * \
                ([self.sym_sc_['Fmin_kN_'] - self.sym_sc_['Fmax_kN_']] * (m - 1) - self.F_cst.T)
            ir_ += m - 1

        if not self.b_ini_done:

            # Power lower bound
            self.lo[ir_:ir_ + m - 1, 0] = [- np.inf] * (m - 1)
            ir_ += m - 1

            # Tire lower bound
            self.lo[ir_:ir_ + m - 1, 0] = [- np.inf] * (m - 1)
            ir_ += m - 1

            # Tire lower bound
            self.lo[ir_:ir_ + m - 1, 0] = [- np.inf] * (m - 1)
            ir_ += m - 1

            # Tire lower bound
            self.lo[ir_:ir_ + m - 1, 0] = [- np.inf] * (m - 1)
            ir_ += m - 1

            # Tire lower bound
            self.lo[ir_:ir_ + m - 1, 0] = [- np.inf] * (m - 1)
            ir_ += m - 1
        else:

            # Move pointer ir_
            ir_ += (m - 1) * 5

        # delta force box lower bound
        # self.lo[ir_:ir_ + m-2, 0] = [self.sym_sc_['dF_kN_neg_'] - self.sym_sc_['dF_kN_pos_']] * (m - 2) - \
        # (self.dF_cst).T

        ################################################################################################################
        # u: - g(x0)
        ################################################################################################################
        ir_ = 0

        # Velocity floor constraint to avoid v < 0 on v__1, ..., v__m-1
        if v_max_cstr is None:

            self.up[ir_:ir_ + m - 2, 0] = [np.inf] * (m - 2)

        # Put hard constraint on v: to be used only when objects appear in optimization horizon from its end
        else:

            self.up[ir_:ir_ + m - 2, 0] = - x0_wh[:-1] + v_max_cstr[1:-1]
        ir_ += m - 2

        # Ceiling on tire slack variables
        self.up[ir_:ir_ + n, 0] = self.sym_sc_['s_v_t_unit_'] * (- x0_s_t + self.sym_sc_['s_v_t_lim_'])
        ir_ += n

        # End velocity upper bound
        self.up[ir_, 0] = - self.v_cst_end
        ir_ += 1

        # --- Don't fill Emergency SQP with F_ini constraint
        if self.sid != 'EmergSQP':

            # F_ini upper bound
            self.up[ir_, 0] = - 0.1 * self.F_ini_cst
            ir_ += 1

            # Force box upper bound
            self.up[ir_:ir_ + m - 2, 0] = - 0.1 * self.F_cst[1:, :].T
            ir_ += m - 2

        else:

            # Force box upper bound
            self.up[ir_:ir_ + m - 1, 0] = - 0.1 * self.F_cst.T
            ir_ += m - 1

        # Power constraint upper bound
        self.up[ir_:ir_ + m - 1, 0] = - 0.01 * self.P_cst.T
        ir_ += m - 1

        # Tire constraint upper bound
        self.up[ir_:ir_ + m - 1, 0] = - self.sym_sc_['tre_cst_w_'] * self.Tre_cst1.T  # tire 1 hard constraint
        ir_ += m - 1

        # Tire constraint upper bound
        self.up[ir_:ir_ + m - 1, 0] = - self.sym_sc_['tre_cst_w_'] * self.Tre_cst2.T  # tire 2 hard constraint
        ir_ += m - 1

        # Tire constraint upper bound
        self.up[ir_:ir_ + m - 1, 0] = - self.sym_sc_['tre_cst_w_'] * self.Tre_cst3.T  # tire 1 hard constraint
        ir_ += m - 1

        # Tire constraint upper bound
        self.up[ir_:ir_ + m - 1, 0] = - self.sym_sc_['tre_cst_w_'] * self.Tre_cst4.T  # tire 2 hard constraint
        ir_ += m - 1

        # Delta-force box upper bound
        # self.up[ir_:ir_+ m-2, 0] = - (self.dF_cst).T  # u = np.append(u, np.array(- (self.dF_pos_cst_).T))
        # ir_ += m-2

        # --- Set flag after first Matrix-initialization
        self.b_ini_done = True

        return P, q, self.Am_csc, self.lo, self.up


if __name__ == '__main__':
    pass
