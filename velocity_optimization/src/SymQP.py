import sympy as sym
import numpy as np
import configparser
from velocity_optimization.src.params_vp_sqp import params_vp_sqp


class SymQP:

    __slots__ = ('m',
                 'n',
                 'sid',
                 'params_path',
                 'sqp_stgs',
                 'sym_sc',
                 'sym_sc_',
                 'E_mat_',
                 'D_mat_',
                 'v',
                 'v_ini',
                 'v_end',
                 's_t',
                 'kappa',
                 'delta_s',
                 'v_max',
                 'ax_max',
                 'ay_max',
                 'F_ini',
                 'P_max',
                 'J_jac',
                 'J_hess',
                 'v_cst',
                 'v_cst_jac',
                 'v_cst_end',
                 'v_cst_end_jac',
                 'F_cst',
                 'F_cst_jac',
                 'F_ini_cst',
                 'F_ini_cst_jac',
                 'dF_cst',
                 'dF_cst_jac',
                 'P_cst',
                 'P_cst_jac',
                 'Tre_cst1',
                 'Tre_cst1_jac',
                 'Tre_cst2',
                 'Tre_cst2_jac',
                 'Tre_cst3',
                 'Tre_cst3_jac',
                 'Tre_cst4',
                 'Tre_cst4_jac',
                 'fJ_jac',
                 'fJ_Hess',
                 'fF_cst',
                 'fF_cst_jac',
                 'fF_ini_cst',
                 'fF_ini_cst_jac',
                 'fdF_cst',
                 'fdF_cst_jac',
                 'fv_cst',
                 'fv_cst_jac',
                 'fv_cst_end',
                 'fv_cst_end_jac',
                 'fP_cst',
                 'fP_cst_jac',
                 'fTre_cst1',
                 'fTre_cst1_jac',
                 'fTre_cst2',
                 'fTre_cst2_jac',
                 'fTre_cst3',
                 'fTre_cst3_jac',
                 'fTre_cst4',
                 'fTre_cst4_jac')

    def __init__(self,
                 m: int,
                 sid: str,
                 params_path: str):

        """
        Python version: 3.5
        Created by: Thomas Herrmann (thomas.herrmann@tum.de)
        Created on: 01.11.2019

        Documentation: Class to construct velocity SQP-optimizer symbolically.

        Inputs:
        m: number of velocity points
        sid: ID of SQP object to be created 'PerfSQP' or 'EmergSQP'
        params_path: absolute path to folder containing config file .ini
        """

        # Number of velocity points
        self.m = m

        # ID ob optimizer object
        self.sid = sid

        # Parameter files path
        self.params_path = params_path

        # SQP settings
        self.sqp_stgs = params_vp_sqp(m=m,
                                      sid=sid,
                                      params_path=self.params_path)[0]

        # Number of slack variables
        self.n = int(np.ceil(m / self.sqp_stgs['slack_every_v']))

        ################################################################################################################
        # --- Get constant numeric values
        ################################################################################################################

        sqp_config = configparser.ConfigParser()
        if not sqp_config.read(self.params_path + 'sqp_config.ini'):
            raise ValueError('Specified sqp config file does not exist or is empty!')

        # --- SQP settings
        if self.sid == 'PerfSQP':
            id_str = 'SOLVER_PERFORMANCE'
        else:
            id_str = 'SOLVER_EMERGENCY'
        sym_sc_ = {'m_t_': sqp_config.getfloat('VEHICLE', 'mass_t'),
                   'Pmax_kW_': sqp_config.getfloat('VEHICLE', 'P_max_kW'),
                   'Fmax_kN_': sqp_config.getfloat('VEHICLE', 'F_max_kN'),
                   'Fmin_kN_': sqp_config.getfloat('VEHICLE', 'F_min_kN'),
                   'axmax_mps2_': sqp_config.getfloat('VEHICLE', 'ax_max_mps2'),
                   'aymax_mps2_': sqp_config.getfloat('VEHICLE', 'ay_max_mps2'),
                   'dF_kN_pos_': sqp_config.getfloat('SOLVER_GENERAL', 'dF_kN_pos'),
                   'dF_kN_neg_': sqp_config.getfloat('SOLVER_GENERAL', 'dF_kN_neg'),
                   'Fini_tol_': sqp_config.getfloat(id_str, 'F_ini_tol'),
                   'c_res_': sqp_config.getfloat('VEHICLE', 'c_res'),
                   'vmin_mps_': sqp_config.getfloat('SOLVER_GENERAL', 'v_min_mps'),
                   's_v_t_lim_': sqp_config.getfloat(id_str, 'slack_var_tire_lim'),
                   's_v_t_unit_': sqp_config.getfloat('SOLVER_GENERAL', 'slack_var_tire_unit'),
                   'dvdv_w_': sqp_config.getfloat(id_str, 'penalty_jerk'),
                   'tre_cst_w_': sqp_config.getfloat(id_str, 'w_tre_constraint'),
                   's_tre_w_lin_': sqp_config.getfloat(id_str, 'penalty_slack_tire_lin'),
                   's_tre_w_quad_': sqp_config.getfloat(id_str, 'penalty_slack_tire_quad')
                   }

        # Assign settings to QP object
        self.sym_sc_ = sym_sc_

        self.E_mat_ = None
        self.D_mat_ = None

        # --- States and parameters
        self.v = None
        self.v_ini = None
        self.v_end = None
        self.s_t = None
        self.kappa = None
        self.delta_s = None
        self.v_max = None
        self.ax_max = None
        self.ay_max = None
        self.F_ini = None
        self.P_max = None
        # Hard-coded parameters
        self.sym_sc = None

        # --- Objective
        self.J_jac = None
        self.J_hess = None

        # --- Constraints
        self.v_cst = None
        self.v_cst_jac = None
        self.v_cst_end = None
        self.v_cst_end_jac = None
        self.F_cst = None
        self.F_cst_jac = None
        self.F_ini_cst = None
        self.F_ini_cst_jac = None
        self.dF_cst = None
        self.dF_cst_jac = None
        self.P_cst = None
        self.P_cst_jac = None
        self.Tre_cst1 = None
        self.Tre_cst1_jac = None
        self.Tre_cst2 = None
        self.Tre_cst2_jac = None
        self.Tre_cst3 = None
        self.Tre_cst3_jac = None
        self.Tre_cst4 = None
        self.Tre_cst4_jac = None

        # --- Lambdified expressions
        self.fJ_jac = None
        self.fJ_Hess = None
        self.fF_cst = None
        self.fF_cst_jac = None
        self.fF_ini_cst = None
        self.fF_ini_cst_jac = None
        self.fdF_cst = None
        self.fdF_cst_jac = None
        self.fv_cst = None
        self.fv_cst_jac = None
        self.fv_cst_end = None
        self.fv_cst_end_jac = None
        self.fP_cst = None
        self.fP_cst_jac = None
        self.fTre_cst1 = None
        self.fTre_cst1_jac = None
        self.fTre_cst2 = None
        self.fTre_cst2_jac = None
        self.fTre_cst3 = None
        self.fTre_cst3_jac = None
        self.fTre_cst4 = None
        self.fTre_cst4_jac = None

        ################################################################################################################
        # Initialize symbolic QP for velocity planning
        ################################################################################################################
        self.init_symbolics()

        ################################################################################################################
        # Substitute constant parameters within symbolic expressions of velocity planner QP
        ################################################################################################################
        self.subs_symbolics()

    def init_symbolics(self):

        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.11.2019

        Documentation: Initializes necessary symbolic expressions to construct velocity optimizer QP.
        """

        ################################################################################################################
        # Dimensions
        ################################################################################################################
        m = self.m
        n = self.n

        ################################################################################################################
        # Matrices for symbolic calculations
        ################################################################################################################

        # Differentiation matrix
        D_mat = sym.Matrix(- np.eye(m) + np.eye(m, k=1))
        # rm last row
        D_mat = sym.Matrix(D_mat[0:m - 1, 0:m])

        # Reduced differentiation matrix
        D_mat_red = sym.Matrix(- np.eye(m - 1) + np.eye(m - 1, k=1))
        # rm last 2 rows and last column
        D_mat_red = sym.Matrix(D_mat_red[0:m - 2, 0:m - 1])

        # m-1 x m matrix with '1' on diagonal
        D_mat_k0 = sym.Matrix(np.eye(m - 1, m))
        # m-1 x m matrix with '1' on first minor diagonal
        D_mat_k1 = sym.Matrix(np.eye(m - 1, m, k=1))

        # m-1 x 1 vector with '1'-entries
        ones_vec_red = sym.ones(m - 1, 1)

        # m x 1 vector with 1, 0, 0, ..., 0-entries
        one_vec = np.zeros((m, 1))
        one_vec[0] = 1

        # m-1 x 1 vector with 1, 0, 0, ..., 0-entries
        one_vec_red = np.zeros((m - 1, 1))
        one_vec_red[0] = 1

        # m x 1 vector with 0, 0, ..., 1-entries
        last_vec = np.zeros((m, 1))
        last_vec[-1] = 1

        # m-1 x m-1 unity matrix
        self.E_mat_ = np.eye(m - 1)
        # Numeric differentiation matrix
        self.D_mat_ = (- np.eye(m) + np.eye(m, k=1))[0:m - 1, 0:m]

        ################################################################################################################
        # Symbolic scalars
        ################################################################################################################
        sym_sc = {'m_t': sym.symbols('m_t'),
                  'Pmax_kW': sym.symbols('Pmax_kW'),
                  'Fmax_kN': sym.symbols('Fmax_kN'),
                  'Fmin_kN': sym.symbols('Fmin_kN'),
                  'axmax_mps2': sym.symbols('axmax_mps2'),
                  'aymax_mps2': sym.symbols('aymax_mps2'),
                  'dF_kN_pos': sym.symbols('dF_kN_pos'),
                  'dF_kN_neg': sym.symbols('dF_kN_neg'),
                  'Fini_tol': sym.symbols('Fini_tol'),
                  'c_res': sym.symbols('c_res'),
                  'dvdv_w': sym.symbols('dvdv_w'),
                  's_tre_w_lin': sym.symbols('s_tre_w_lin'),
                  's_tre_w_quad': sym.symbols('s_tre_w_quad'),
                  's_v_t_unit': sym.symbols('s_v_t_unit')
                  }

        # Initial velocity constraint [m/s]
        v_ini = sym.symbols('v_ini')
        # End velocity constraint in optimization horizon [m/s]
        v_end = sym.symbols('v_end')
        # Initial force constraint [kN]
        F_ini = sym.symbols('F_ini')

        # --- Assign to class
        self.sym_sc = sym_sc
        self.v_ini = v_ini
        self.v_end = v_end
        self.F_ini = F_ini

        ################################################################################################################
        # Symbolic states and parameter vectors
        ################################################################################################################

        # --- Create vector containing velocity optimization variables
        v = sym.Matrix(sym.symbols('v0:%d' % m)).T
        # --- Create vector containing slack tire optimization variables
        s_t = sym.Matrix(sym.symbols('s_t0:%d' % n)).T
        # --- Create vector containing all kappa parameters
        kappa = sym.Matrix(sym.symbols('kappa0:%d' % m)).T
        # --- Create vector containing all delta_s parameters
        delta_s = sym.Matrix(sym.symbols('delta_s0:%d' % (m - 1))).T
        # --- Create vector containing all v_max parameters
        v_max = sym.Matrix(sym.symbols('v_max0:%d' % m)).T
        # --- Create vector containing all ax_max parameters
        ax_max = sym.Matrix(sym.symbols('ax_max0:%d' % m)).T
        # --- Create vector containing all ay_max parameters
        ay_max = sym.Matrix(sym.symbols('ay_max0:%d' % m)).T
        # --- Create vector containing all P_max parameters
        P_max = sym.Matrix(sym.symbols('P_max0:%d' % (m - 1))).T

        print('# --- Optimization vector velocity v --- #')
        sym.pprint(v)
        print('# --- Optimization vector slack s_t --- #')
        sym.pprint(s_t)
        print('# --- Parameter vector kappa --- #')
        sym.pprint(kappa)
        print('# --- Parameter vector delta_s --- #')
        sym.pprint(delta_s)
        print('# --- Parameter vector v_max --- #')
        sym.pprint(v_max)
        print('# --- Parameter vector ax_max --- #')
        sym.pprint(ax_max)
        print('# --- Parameter vector ay_max --- #')
        sym.pprint(ay_max)
        print('# --- Parameter vector P_max --- #')
        sym.pprint(P_max)

        ################################################################################################################
        # Symbolic objective
        ################################################################################################################

        v_red = sym.eye(m - 1, m) * v.T

        v_red_inv = v_red.applyfunc(lambda x: 1 / x)
        print('# --- Elementwise inverse of v_red --- #')
        sym.pprint(v_red_inv.T)

        if self.sqp_stgs['b_var_friction']:
            ax_max_inv = ax_max.applyfunc(lambda x: 1 / x)
            ay_max_inv = ay_max.applyfunc(lambda x: 1 / x)
        else:
            ax_max_inv = None
            ay_max_inv = None

        ################################################################################################################
        # Define the objective function J
        ################################################################################################################
        # Objective using J = min. sum(v - v_max) ** 2
        J = (v - v_max).applyfunc(lambda x: sym.Pow(x, 2)) * sym.ones(m, 1)

        # --- Include dvdv (jerk-regularization) in objective function J
        dvdv = sym.MatMul(D_mat[:-1, :-1], sym.MatMul(D_mat, v.T)).doit()
        dvdv_squ = dvdv.applyfunc(lambda x: x ** 2)

        # --- Add contribution of dvdv_squ (jerk)
        J = J + \
            self.sym_sc['dvdv_w'] * sym.ones(1, m - 2) * dvdv_squ

        # --- Add contribution of tire slack variables to objective (square and linear)
        J = J + \
            self.sym_sc['s_tre_w_lin'] * \
            s_t * self.sym_sc['s_v_t_unit'] * sym.ones(n, 1) + \
            self.sym_sc['s_tre_w_quad'] * \
            s_t.applyfunc(lambda x: sym.Pow(self.sym_sc['s_v_t_unit'] * x, 2)) * sym.ones(n, 1)

        print('# --- Objective J --- #')
        sym.pprint(J, wrap_line=False)

        # --- Concatenate all optimization variables
        o = sym.BlockMatrix([[v, s_t]]).as_explicit()

        # --- Derivative of J after velocity optimization variables as well as slack optimization variables
        J_jac = J.jacobian(o)
        print('# --- Jacobian of J --- #')
        sym.pprint(J_jac, wrap_line=False)

        # --- Derivative of J after velocity optimization variables as well as slack optimization variables
        print('# --- Hessian of J --- #')
        J_hess = sym.hessian(J, o)
        sym.pprint(J_hess, wrap_line=False)

        ################################################################################################################
        # Symbolic constraints
        ################################################################################################################
        p_delta_s_inv = delta_s.applyfunc(lambda x: 1 / x)

        # --- Create diagonal-matrix from reduced velocity-vector
        i = 0
        v_mat_red = sym.eye(m - 1, m - 1)
        for e in v_red:
            v_mat_red[i, i] = e
            i += 1
        # Inverse of delta-time vector
        dt_inv = sym.MatMul(v_mat_red, p_delta_s_inv.T).doit()

        # --- Create diagonal-matrix from inverse delta-time vector
        dt_mat_inv = sym.eye(m - 1, m - 1)
        i = 0
        for e in dt_inv.T:
            dt_mat_inv[i, i] = e
            i += 1

        # --- Acceleration for tire constraints
        acc = sym.zeros(m - 1, 1)
        a1 = sym.MatMul(D_mat_k1, v.T).doit().applyfunc(lambda x: x ** 2)
        a2 = sym.MatMul(D_mat_k0, v.T).doit().applyfunc(lambda x: x ** 2)
        delta_2s_inv = (2 * delta_s).applyfunc(lambda x: 1 / x)
        i = 0
        for e in delta_2s_inv:
            acc[i] = (a1[i] - a2[i]) * e
            i += 1

        # --- Calculate from powertrain applied force [kN]
        F_p = sym.MatMul(sym_sc['m_t'], acc) + \
            sym.MatMul(sym_sc['c_res'], sym.Matrix(v.applyfunc(lambda x: x ** 2)[0:m - 1])) * 0.001

        # --- Force initial hard constraint [kN]
        F_ini_cst = sym.MatMul(sym.Matrix(one_vec_red).T, F_p).doit() - sym.MatMul(F_ini, sym.ones(1, 1)) - \
            sym.MatMul(sym_sc['Fini_tol'], sym.ones(1, 1))

        # --- Jacobian from Force initial constraint
        F_ini_cst_jac = F_ini_cst.jacobian(o)

        # --- Velocity initial [m/s]
        v_cst = sym.MatMul(sym.Matrix(one_vec).T, v.T).doit() - sym.MatMul(v_ini, sym.ones(1, 1))

        # --- Jacobian from velocity-initial-constraint
        v_cst_jac = v_cst.jacobian(o)

        # --- Upper velocity end constraint [m/s]
        v_cst_end = sym.MatMul(sym.Matrix(last_vec).T, v.T).doit() - sym.MatMul(v_end, sym.ones(1, 1))

        # --- Jacobian from velocity-end-constraint
        v_cst_end_jac = v_cst_end.jacobian(o)

        # --- Force box hard constraint [kN]
        F_cst = (F_p - sym.MatMul(sym_sc['Fmax_kN'], sym.Matrix(ones_vec_red))).doit()

        # Jacobian from force-constraint
        F_cst_jac = F_cst.jacobian(o)

        # --- Delta-force box hard constraints (actuators) [kN]
        dF_cst = sym.MatMul(D_mat_red, F_p) - sym.MatMul(sym.ones(m - 2, 1), sym_sc['dF_kN_pos'])

        # --- Jacobian from Delta-force box
        dF_cst_jac = dF_cst.jacobian(o)

        # --- Power hard constraint [kW] as diagonal matrix
        F_p_mat = sym.eye(m - 1)
        for i in range(0, m - 1):
            F_p_mat[i, i] = F_p[i, 0]

        # --- Check for variable power limitation
        if self.sqp_stgs['b_var_power']:
            P_cst = (sym.MatMul(F_p_mat, v_red) - P_max.T).doit()
        else:
            P_cst = sym.MatMul(F_p_mat, v_red) - sym.MatMul(sym_sc['Pmax_kW'], sym.Matrix(ones_vec_red))

        # --- Jacobian of power constraint
        P_cst_jac = P_cst.jacobian(o)

        # --- Tire hard constraints [-]
        kappa_mat = sym.eye(m - 1)
        for i in range(0, m - 1):
            kappa_mat[i, i] = kappa[i]
        # Velocities squared
        v_squ = v.applyfunc(lambda x: x ** 2)
        # Cut last row of velocity diagonal matrix
        v_squ = sym.MatMul(sym.eye(m - 1, m), v_squ.T).doit()

        # Variable friction matrices with inverse entries of max. acceleration limits [s^2/m]
        if self.sqp_stgs['b_var_friction']:
            ax_max_mat_inv = sym.eye(m - 1)
            ay_max_mat_inv = sym.eye(m - 1)
            for i in range(0, m - 1):
                ax_max_mat_inv[i, i] = ax_max_inv[i]
                ay_max_mat_inv[i, i] = ay_max_inv[i]
        else:
            ax_max_mat_inv = None
            ay_max_mat_inv = None

        # Calculate norm values of accelerations [-]
        if self.sqp_stgs['b_var_friction']:
            ax_norm = sym.MatMul((F_p / sym_sc['m_t']).T, ax_max_mat_inv).T.doit()
            ay_norm = sym.MatMul(sym.MatMul(kappa_mat, v_squ).T, ay_max_mat_inv).T.doit()
        else:
            ax_norm = sym.MatMul(F_p / sym_sc['m_t'], 1 / sym_sc['axmax_mps2']).doit()
            ay_norm = sym.MatMul(sym.MatMul(kappa_mat, v_squ), 1 / sym_sc['aymax_mps2']).doit()

        # --- Adding 4 tire constraints to implement abs()-function, converted to mutable matrices
        Tre_cst1 = (ax_norm + ay_norm - ones_vec_red).doit().as_mutable()
        Tre_cst2 = (- ax_norm + ay_norm - ones_vec_red).doit().as_mutable()
        Tre_cst3 = (- ax_norm - ay_norm - ones_vec_red).doit().as_mutable()
        Tre_cst4 = (ax_norm - ay_norm - ones_vec_red).doit().as_mutable()

        # --- Adding tire slack variables to tire hard constraints
        # counter for tire constraints
        i = 0
        # counter variable for slacks
        j = 0
        for _ in Tre_cst1:

            # if i reaches switching point, go to next slack variable and apply to subsequent tire entries
            if np.mod(i, self.sqp_stgs['slack_every_v']) == 0 and i >= self.sqp_stgs['slack_every_v']:
                j += 1
            Tre_cst1[i] -= self.sym_sc['s_v_t_unit'] * s_t[j]
            Tre_cst2[i] -= self.sym_sc['s_v_t_unit'] * s_t[j]
            Tre_cst3[i] -= self.sym_sc['s_v_t_unit'] * s_t[j]
            Tre_cst4[i] -= self.sym_sc['s_v_t_unit'] * s_t[j]
            i += 1

        # Convert tire constraints back to immutable matrices
        Tre_cst1 = sym.ImmutableMatrix(Tre_cst1)
        Tre_cst2 = sym.ImmutableMatrix(Tre_cst2)
        Tre_cst3 = sym.ImmutableMatrix(Tre_cst3)
        Tre_cst4 = sym.ImmutableMatrix(Tre_cst4)

        # Jacobian of tire constraints
        Tre_cst1_jac = Tre_cst1.jacobian(o)
        Tre_cst2_jac = Tre_cst2.jacobian(o)
        Tre_cst3_jac = Tre_cst3.jacobian(o)
        Tre_cst4_jac = Tre_cst4.jacobian(o)

        ################################################################################################################
        # Assign state vector
        ################################################################################################################
        self.v = v
        self.s_t = s_t
        self.kappa = kappa
        self.delta_s = delta_s
        self.v_max = v_max
        self.ax_max = ax_max
        self.ay_max = ay_max
        self.P_max = P_max

        ################################################################################################################
        # Assign objective
        ################################################################################################################
        self.J_jac = J_jac
        self.J_hess = J_hess

        ################################################################################################################
        # Assign constraints
        ################################################################################################################
        self.v_cst = v_cst
        self.v_cst_jac = v_cst_jac
        self.v_cst_end = v_cst_end
        self.v_cst_end_jac = v_cst_end_jac
        self.F_cst = F_cst
        self.F_cst_jac = F_cst_jac
        self.F_ini_cst = F_ini_cst
        self.F_ini_cst_jac = F_ini_cst_jac
        self.dF_cst = dF_cst
        self.dF_cst_jac = dF_cst_jac
        self.P_cst = P_cst
        self.P_cst_jac = P_cst_jac
        self.Tre_cst1 = Tre_cst1
        self.Tre_cst1_jac = Tre_cst1_jac
        self.Tre_cst2 = Tre_cst2
        self.Tre_cst2_jac = Tre_cst2_jac
        self.Tre_cst3 = Tre_cst3
        self.Tre_cst3_jac = Tre_cst3_jac
        self.Tre_cst4 = Tre_cst4
        self.Tre_cst4_jac = Tre_cst4_jac

    def subs_symbolics(self):

        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.11.2019

        Documentation: Substitutes hard coded parameters in symbolic expressions.
        """

        v = self.v
        s_t = self.s_t
        kappa = self.kappa
        delta_s = self.delta_s
        v_max = self.v_max
        ax_max = self.ax_max
        ay_max = self.ay_max
        P_max = self.P_max

        v_ini = self.v_ini
        v_end = self.v_end
        F_ini = self.F_ini
        F_ini_tol = self.sym_sc['Fini_tol']
        dvdv_w = self.sym_sc['dvdv_w']
        s_tre_w_lin = self.sym_sc['s_tre_w_lin']
        s_tre_w_quad = self.sym_sc['s_tre_w_quad']
        s_v_t_unit = self.sym_sc['s_v_t_unit']

        # --- Force
        self.F_cst = self.subs_syms(self.F_cst)
        self.F_cst_jac = self.subs_syms(self.F_cst_jac)

        # --- Delta-force box
        self.dF_cst = self.subs_syms(self.dF_cst)
        self.dF_cst_jac = self.subs_syms(self.dF_cst_jac)

        # --- Force initial
        self.F_ini_cst = self.subs_syms(self.F_ini_cst)
        self.F_ini_cst_jac = self.subs_syms(self.F_ini_cst_jac)

        # --- Power
        self.P_cst = self.subs_syms(self.P_cst)
        self.P_cst_jac = self.subs_syms(self.P_cst_jac)

        # --- Tire
        self.Tre_cst1 = self.subs_syms(self.Tre_cst1)
        self.Tre_cst1_jac = self.subs_syms(self.Tre_cst1_jac)
        self.Tre_cst2 = self.subs_syms(self.Tre_cst2)
        self.Tre_cst2_jac = self.subs_syms(self.Tre_cst2_jac)
        self.Tre_cst3 = self.subs_syms(self.Tre_cst3)
        self.Tre_cst3_jac = self.subs_syms(self.Tre_cst3_jac)
        self.Tre_cst4 = self.subs_syms(self.Tre_cst4)
        self.Tre_cst4_jac = self.subs_syms(self.Tre_cst4_jac)

        ################################################################################################################
        # Create function handles from symbolic expressions with necessary free parameters
        ################################################################################################################

        # --- Lambdify objective
        self.fJ_jac = \
            sym.lambdify([list(v.values()), list(s_t.values()), v_max, dvdv_w, s_tre_w_lin, s_tre_w_quad, s_v_t_unit],
                         self.J_jac, 'numpy')

        self.fJ_Hess = sym.lambdify([list(v.values()), dvdv_w, s_tre_w_lin, s_tre_w_quad, s_v_t_unit],
                                    self.J_hess, 'numpy')

        # --- Lambdify constraints
        # Force box
        self.fF_cst = sym.lambdify([list(v.values()), list(delta_s.values())], self.F_cst, 'numpy')
        self.fF_cst_jac = sym.lambdify([list(v.values()), list(delta_s.values())], self.F_cst_jac, 'numpy')

        # Force initial
        self.fF_ini_cst = sym.lambdify([list(v.values()), list(delta_s.values()), F_ini, F_ini_tol],
                                       self.F_ini_cst, 'numpy')
        self.fF_ini_cst_jac = sym.lambdify([list(v.values()), list(delta_s.values())],
                                           self.F_ini_cst_jac, 'numpy')

        # Initial velocity
        self.fv_cst = sym.lambdify([list(v.values()), v_ini], self.v_cst, 'numpy')
        self.fv_cst_jac = sym.lambdify([list(v.values())], self.v_cst_jac, 'numpy')

        # End velocity
        self.fv_cst_end = sym.lambdify([list(v.values()), v_end], self.v_cst_end, 'numpy')
        self.fv_cst_end_jac = sym.lambdify([], self.v_cst_end_jac, 'numpy')

        # Power
        if self.sqp_stgs['b_var_power']:
            self.fP_cst = sym.lambdify([list(v.values()), list(delta_s.values()), list(P_max.values())],
                                       self.P_cst, 'numpy')
        else:
            self.fP_cst = sym.lambdify([list(v.values()), list(delta_s.values())], self.P_cst, 'numpy')
        self.fP_cst_jac = sym.lambdify([list(v.values()), list(delta_s.values())], self.P_cst_jac, 'numpy')

        # Tires
        if self.sqp_stgs['b_var_friction']:
            self.fTre_cst1 = \
                sym.lambdify([list(v.values()), list(s_t.values()), list(kappa.values()), list(delta_s.values()),
                              list(ax_max.values()), list(ay_max.values())],
                             self.Tre_cst1, 'numpy')
            self.fTre_cst1_jac = \
                sym.lambdify([list(v.values()), list(s_t.values()), list(kappa.values()), list(delta_s.values()),
                              list(ax_max.values()), list(ay_max.values())],
                             self.Tre_cst1_jac, 'numpy')
            self.fTre_cst2 = \
                sym.lambdify([list(v.values()), list(s_t.values()), list(kappa.values()), list(delta_s.values()),
                              list(ax_max.values()), list(ay_max.values())],
                             self.Tre_cst2, 'numpy')
            self.fTre_cst2_jac = \
                sym.lambdify([list(v.values()), list(s_t.values()), list(kappa.values()), list(delta_s.values()),
                              list(ax_max.values()), list(ay_max.values())],
                             self.Tre_cst2_jac, 'numpy')
            self.fTre_cst3 = \
                sym.lambdify([list(v.values()), list(s_t.values()), list(kappa.values()), list(delta_s.values()),
                              list(ax_max.values()), list(ay_max.values())],
                             self.Tre_cst3, 'numpy')
            self.fTre_cst3_jac = \
                sym.lambdify([list(v.values()), list(s_t.values()), list(kappa.values()), list(delta_s.values()),
                              list(ax_max.values()), list(ay_max.values())],
                             self.Tre_cst3_jac, 'numpy')
            self.fTre_cst4 = \
                sym.lambdify([list(v.values()), list(s_t.values()), list(kappa.values()), list(delta_s.values()),
                              list(ax_max.values()), list(ay_max.values())],
                             self.Tre_cst4, 'numpy')
            self.fTre_cst4_jac = \
                sym.lambdify([list(v.values()), list(s_t.values()), list(kappa.values()), list(delta_s.values()),
                              list(ax_max.values()), list(ay_max.values())],
                             self.Tre_cst4_jac, 'numpy')

        else:
            self.fTre_cst1 = \
                sym.lambdify([list(v.values()), list(s_t.values()), list(kappa.values()), list(delta_s.values())],
                             self.Tre_cst1, 'numpy')
            self.fTre_cst1_jac = \
                sym.lambdify([list(v.values()), list(s_t.values()), list(kappa.values()), list(delta_s.values())],
                             self.Tre_cst1_jac, 'numpy')
            self.fTre_cst2 = \
                sym.lambdify([list(v.values()), list(s_t.values()), list(kappa.values()), list(delta_s.values())],
                             self.Tre_cst2, 'numpy')
            self.fTre_cst2_jac = \
                sym.lambdify([list(v.values()), list(s_t.values()), list(kappa.values()), list(delta_s.values())],
                             self.Tre_cst2_jac, 'numpy')
            self.fTre_cst3 = \
                sym.lambdify([list(v.values()), list(s_t.values()), list(kappa.values()), list(delta_s.values())],
                             self.Tre_cst3, 'numpy')
            self.fTre_cst3_jac = \
                sym.lambdify([list(v.values()), list(s_t.values()), list(kappa.values()), list(delta_s.values())],
                             self.Tre_cst3_jac, 'numpy')
            self.fTre_cst4 = \
                sym.lambdify([list(v.values()), list(s_t.values()), list(kappa.values()), list(delta_s.values())],
                             self.Tre_cst4, 'numpy')
            self.fTre_cst4_jac = \
                sym.lambdify([list(v.values()), list(s_t.values()), list(kappa.values()), list(delta_s.values())],
                             self.Tre_cst4_jac, 'numpy')

        # Delta-force box
        self.fdF_cst = sym.lambdify([list(v.values()), list(delta_s.values())], self.dF_cst, 'numpy')
        self.fdF_cst_jac = sym.lambdify([list(v.values()), list(delta_s.values())],
                                        self.dF_cst_jac, 'numpy')

    def subs_syms(self,
                  arg) -> sym.ImmutableDenseMatrix:

        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.11.2019

        Documentation: Substitutes symbols by their respective numeric value.

        Inputs:
        arg: symbolic expression

        Outputs:
        arg: symbolic expression with replaced hard-coded parameters
        """

        for sym_scalar in self.sym_sc.values():
            arg = arg.subs(sym_scalar, self.sym_sc_[sym_scalar.name + '_'])

        return arg


if __name__ == '__main__':
    pass
