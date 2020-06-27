try:
    import casadi as cs
except ImportError:
    print('Warning: No module CasADi found. Not necessary on car but for development.')
import numpy as np
import time
import configparser


class VOptIPOPT:

    __slots__ = ('m',
                 'n',
                 'sid',
                 'slack_every_v',
                 'params_path',
                 'b_warm',
                 'trj',
                 'solver',
                 'unit_eps_tre',
                 'x',
                 'x0',
                 'lbx',
                 'ubx',
                 'g',
                 'lbg',
                 'ubg',
                 'J',
                 'lam_x0',
                 'lam_g0')

    def __init__(self,
                 m: int,
                 sid: str,
                 slack_every_v: int,
                 params_path: str,
                 b_warm: bool = False):
        """
        Python version: 3.5
        Created by: Thomas Herrmann (thomas.herrmann@tum.de)
        Created on: 01.01.2020

        Documentation: Class to optimize a velocity profile for a given path using the solver IPOPT interfaced by
        CasADi. This class supports only constant power limitations but variable friction.

        Inputs:
        m: number of optimization velocity points
        sid: ID of optimizer object 'EmergSQP' or 'PerfSQP'
        slack_every_v: how many velocity variables used one slack variable
        b_warm: allow warm-start of IPOPT solver
        """

        self.m = m
        self.n = m - 1
        self.sid = sid
        self.slack_every_v = slack_every_v
        self.unit_eps_tre = 0
        self.b_warm = b_warm
        self.trj = None
        self.solver = None

        self.x = None
        self.x0 = None
        self.lbx = None
        self.ubx = None
        self.g = None
        self.lbg = None
        self.ubg = None
        self.J = None
        self.lam_x0 = None
        self.lam_g0 = None

        self.params_path = params_path

        self.sol_init()

    def sol_init(self):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.01.2020

        Documentation: Builds optimization problem using CasADi modeling language.
        """

        ################################################################################################################
        # Dimensions
        ################################################################################################################
        m = self.m
        n = self.n
        b_warm = self.b_warm

        opt_config = configparser.ConfigParser()
        if not opt_config.read(self.params_path + 'sqp_config.ini'):
            raise ValueError('Specified cost config file does not exist or is empty!')

        ################################################################################################################
        # Affine Expressions
        ################################################################################################################
        if self.sid == 'PerfSQP':
            set_sec = 'SOLVER_PERFORMANCE'
        elif self.sid == 'EmergSQP':
            set_sec = 'SOLVER_EMERGENCY'
        else:
            set_sec = None

        # Matrices
        one_vec = np.zeros((m, 1))
        one_vec[0] = 1  # Vector [1,0,...,0] mx1

        oneone_vec_short = np.ones((n, 1))  # Vector [1,...,1] nx1

        Mat_one_val = np.zeros((n, m))
        Mat_one_val[0, 0] = 1  # Matrix [1,0,...,0;0,...,0;0,...,0] nxm

        # Hard coded params --------------------------------------------------------------------------------------------
        # Vehicle parameters
        # max. power of both e-machines [kW]
        P_max = opt_config.getfloat('VEHICLE', 'P_max_kW')
        # max. tractional force [kN]
        F_max = opt_config.getfloat('VEHICLE', 'F_max_kN')
        # max. braking force [kN]
        F_min = opt_config.getfloat('VEHICLE', 'F_min_kN')
        # vehicle mass [t]
        m_t = opt_config.getfloat('VEHICLE', 'mass_t')

        # Input params: for changing online without rebuilding problem -------------------------------------------------

        eps_tre_max = opt_config.getfloat(set_sec, 'slack_var_tire_lim')

        # Force tolerance on initial point [kN]
        tol_F_cst_ini = opt_config.getfloat(set_sec, 'F_ini_tol')

        # air resistance coefficient
        c_res = opt_config.getfloat('VEHICLE', 'c_res')

        # Units
        unit_eps_tre = opt_config.getfloat('SOLVER_GENERAL', 'slack_var_tire_unit')
        self.unit_eps_tre = unit_eps_tre

        # Weights
        s_eps_tre_lin = opt_config.getfloat(set_sec, 'penalty_slack_tire_lin')
        s_eps_tre_quad = opt_config.getfloat(set_sec, 'penalty_slack_tire_quad')
        penalty_jerk = opt_config. getfloat(set_sec, 'penalty_jerk')

        # --- Variable friction
        if opt_config. getfloat(set_sec, 'b_var_friction'):
            ax_max = None
            ay_max = None
        # --- Constant friction
        else:
            ax_max = opt_config.getfloat('VEHICLE', 'ax_max_mps2')  # max a_x
            ay_max = opt_config.getfloat('VEHICLE', 'ay_max_mps2')  # max a_y

        # Empty NLP
        x = []
        x0 = []
        lbx = []
        ubx = []

        g = []
        lbg = []
        ubg = []

        J = 0

        ################################################################################################################
        # Optimization Variables
        ################################################################################################################
        # Velocity [m/s]
        v = cs.SX.sym('v', m)

        # Tire slack variables epsilon [-]
        eps_tre = cs.SX.sym('eps_tre', int(np.ceil(m / self.slack_every_v)))

        ################################################################################################################
        # Online Parameters
        ################################################################################################################
        # Initial velocity [m/s]
        v_ini_param = cs.SX.sym('v_ini_param', 1)

        # kappa profile [rad/m]
        kappa_param = cs.SX.sym('kappa_param', m)

        # delta s [m]
        delta_s_param = cs.SX.sym('delta_s_param', m - 1)

        # max. velocity [m/s]
        v_max_param = cs.SX.sym('v_max_param', m)

        # initial force constraint [kN]
        F_cst_ini_param = cs.SX.sym('F_cst_ini_param', 1)

        # end velocity in opt. horizon [m/s]
        v_end_param = cs.SX.sym('v_end_param', 1)

        if ax_max is None:
            ax_max = cs.SX.sym('ax_max', m - 1)
            ay_max = cs.SX.sym('ay_max', m - 1)

        ################################################################################################################
        # Constraints
        ################################################################################################################
        # acceleration [m/s^2]
        acc = (v[1:] ** 2 - v[: - 1] ** 2) / (2 * delta_s_param)

        # --- Powertrain force [kN]
        F_p = m_t * acc + \
            c_res * (v ** 2)[0: m - 1] * 0.001

        # --- Force
        # F_min <= F_p__1, ..., F_p__n <= F_max, first point in F excluded as it is F_initial
        F_cst = F_p[1:]
        g.append(F_cst)
        ubg.append([F_max] * (n - 1))
        lbg.append([F_min] * (n - 1))

        # --- Force initial: only in performance solver
        if self.sid == 'PerfSQP':
            F_cst_ini = F_p[0] - F_cst_ini_param
            g.append(F_cst_ini)
            ubg.append([tol_F_cst_ini])
            lbg.append([- tol_F_cst_ini])

        # --- Power
        # P_max * [v_mps__1, ... v_mps__n] >= [F_p__1, ..., F_p__n]
        # 0 >= F_p__i - P_max * 1/v_mps__i >= - inf
        pwr_cst = F_p * v[:-1]
        g.append(pwr_cst)
        ubg.append([P_max] * n)
        lbg.append([-np.inf] * n)

        # --- Tires
        # (| a_x_mps2 | __i / ax_max + a | a_y_mps2 | __i / ay_max - 1__i - eps <= 0__i)
        # tire_cst = 1.0 * \
        #    (1 / ax_max * 1 / m_t * F_p) ** 2 + 1.0 * (1 / ay_max * mtimes(d, (v ** 2))) ** 2
        tire_cst1 = (F_p / m_t) / ax_max + (kappa_param * (v ** 2))[: - 1] / ay_max - \
            oneone_vec_short
        tire_cst2 = (F_p / m_t) / ax_max - (kappa_param * (v ** 2))[: - 1] / ay_max - \
            oneone_vec_short
        tire_cst3 = - (F_p / m_t) / ax_max - (kappa_param * (v ** 2))[: - 1] / ay_max - \
            oneone_vec_short
        tire_cst4 = - (F_p / m_t) / ax_max + (kappa_param * (v ** 2))[: - 1] / ay_max - \
            oneone_vec_short

        i = 0
        # counter variable for slacks
        j = 0
        for tre in range(tire_cst1.shape[0]):
            # if i reaches switching point, go to next slack variable and apply to following tire entries
            if np.mod(i, self.slack_every_v) == 0 and i >= self.slack_every_v:
                j += 1
            tire_cst1[i] -= unit_eps_tre * eps_tre[j]
            tire_cst2[i] -= unit_eps_tre * eps_tre[j]
            tire_cst3[i] -= unit_eps_tre * eps_tre[j]
            tire_cst4[i] -= unit_eps_tre * eps_tre[j]
            i += 1

        g.append(tire_cst1)
        ubg.append([0] * n)
        lbg.append([- np.inf] * n)
        g.append(tire_cst2)
        ubg.append([0] * n)
        lbg.append([- np.inf] * n)
        g.append(tire_cst3)
        ubg.append([0] * n)
        lbg.append([- np.inf] * n)
        g.append(tire_cst4)
        ubg.append([0] * n)
        lbg.append([- np.inf] * n)

        # --- Initial values
        # v_mps__0 == v_ini_param
        g.append(v_ini_param - v[0])
        ubg.append([0])
        lbg.append([0])

        # --- End value velocity
        # v_mps__N-1 == v_end_param
        g.append(v[-1] - v_end_param)
        ubg.append([0])
        lbg.append([- np.inf])

        # --- Velocity
        # 0 <= [v_mps__1, ...,v_mps__m] <= inf
        v_cst = v
        g.append(v_cst)
        ubg.append([np.inf] * m)
        lbg.append([0] * m)

        # inf >= [eps_tre__1,...,eps_tre__m] >= 0
        eps_tre_cst = unit_eps_tre * eps_tre
        g.append(eps_tre_cst)
        lbg.append([0] * int(np.ceil(m / self.slack_every_v)))
        ubg.append([unit_eps_tre * eps_tre_max] * int(np.ceil(m / self.slack_every_v)))

        ################################################################################################################
        # Objective function
        ################################################################################################################
        # sum__i (d_s_m__i * 1/v__i + s_eps_vel * eps_vel_mps__i + s_eps_dF * (dF__i)**2 + s_eps_tre * eps_tre__i)

        J = cs.sum1((v - v_max_param) ** 2) + \
            penalty_jerk * cs.sum1((cs.diff(cs.diff(v))) ** 2) + \
            s_eps_tre_lin * cs.sum1(unit_eps_tre * eps_tre) + \
            s_eps_tre_quad * cs.sum1((unit_eps_tre * eps_tre) ** 2)

        ################################################################################################################
        # Solver stuff
        ################################################################################################################
        # Initialization of optimization variables, OVERRIDDEN during usage
        x0.append([20] * m)  # initial guess for velocity
        x0.append([0] * int(np.ceil(m / self.slack_every_v)))  # initial guess for slack variable tire-constraints

        x0 = np.concatenate(x0)

        # Formatting lower and upper bounds of optimization variables if available
        try:
            lbx = np.concatenate(lbx)
            ubx = np.concatenate(ubx)
        except ValueError:
            pass

        # Formatting constraints
        g = cs.vertcat(*g)
        lbg = np.concatenate(lbg)
        ubg = np.concatenate(ubg)

        # --- Formatting parameter vector for optimization
        # Constant friction
        if isinstance(ax_max, float):
            param_vec = cs.vertcat(v_ini_param,
                                   kappa_param,
                                   delta_s_param,
                                   v_max_param,
                                   F_cst_ini_param,
                                   v_end_param)
        # Variable friction
        else:
            param_vec = cs.vertcat(v_ini_param,
                                   kappa_param,
                                   delta_s_param,
                                   v_max_param,
                                   F_cst_ini_param,
                                   v_end_param,
                                   ax_max,
                                   ay_max)

        # Formatting optimization vector for optimization
        x = cs.vertcat(v, eps_tre)  # create optimization variables vector

        nlp = {'x': x, 'p': param_vec, 'f': J, 'g': g}

        opts_IPOPT = {"expand": False,
                      "ipopt.print_level": 0,  # 0
                      "ipopt.max_iter": 1000,
                      "ipopt.tol": 1e-2,
                      "ipopt.nlp_scaling_method": 'none',
                      "verbose": False,  # False
                      }

        opts_IPOPT_warm = {"expand": False,
                           "ipopt.print_level": 0,  # 0
                           "ipopt.max_iter": 1000,
                           "ipopt.tol": 1e-2,
                           "ipopt.nlp_scaling_method": 'none',
                           "verbose": False,
                           "ipopt.warm_start_init_point": "yes",  # warm start (yes/no)
                           }

        # Create solver
        if b_warm:
            self.solver = cs.nlpsol('solver', 'ipopt', nlp, opts_IPOPT_warm)
            self.lam_x0 = cs.DM([0] * m)
            self.lam_g0 = cs.DM([0] * m)

        else:
            self.solver = cs.nlpsol('solver', 'ipopt', nlp, opts_IPOPT)

        self.trj = cs.Function('trj',
                               [x, param_vec],
                               [F_p, tire_cst1, acc, pwr_cst],
                               ['x', 'kappa_param'],
                               ['F_p', 'tire_cst', 'acc', 'pwr_cst'])

        self.x = x
        self.x0 = x0
        self.lbx = lbx
        self.ubx = ubx
        self.g = g
        self.lbg = lbg
        self.ubg = ubg
        self.J = 0

    def calc_v_ipopt(self,
                     v_ini: np.ndarray,
                     kappa: np.ndarray,
                     delta_s: np.ndarray,
                     v_max: np.ndarray,
                     F_ini: float,
                     v_end: float,
                     x0_v: np.ndarray,
                     x0_s_t: np.ndarray,
                     ax_max: np.ndarray,
                     ay_max: np.ndarray) -> tuple:

        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.01.2020

        Documentation: Builds parameter and initialization vectors for IPOPT solver

        Inputs:
        v_ini: initial velocity [m/s]
        kappa: curvature profile [rad/m]
        delta_s: discretization steplength [m]
        v_max: max. velocity [m/s]
        F_ini: initial force constraint [kN]
        v_end: end velocity within optimization horizon [m/s]
        x0_v: initial velocity guess [m/s]
        x0_s_t: initial guess for slack variables on tires [-]
        ax_max: max. longitudinal acceleration limit [m/s^2]
        ay_max: max. lateral acceleration limit [m/s^2]

        Outputs:
        sol: optimized IPOPT output
        param_vec: variable parameter vector
        dt_ipopt: IPOPT runtimed [ms]
        """

        # --- Parameter vector for NLP
        param_vec = list()

        param_vec.append([v_ini])
        param_vec.append(kappa)
        param_vec.append(delta_s)
        param_vec.append(v_max)
        param_vec.append([F_ini])
        param_vec.append([v_end])
        if ax_max is not None:
            param_vec.append(ax_max)
            param_vec.append(ay_max)

        # Conversion of format
        param_vec = np.concatenate(param_vec)

        # --- Initial guess for NLP
        x0_nlp = list()
        x0_nlp.append(x0_v)
        x0_nlp.append(x0_s_t)

        # Conversion of format
        x0_nlp = np.concatenate(x0_nlp)

        ################################################################################################################
        # Solve NLP
        ################################################################################################################
        sol, dt_ipopt = self.solve(x0=x0_nlp,
                                   param_vec=param_vec)

        return sol, param_vec, dt_ipopt

    def solve(self,
              x0: np.ndarray,
              param_vec: np.ndarray) -> tuple:
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.01.2020

        Documentation: Solves velocity opt. problem using IPOPT

        Inputs:
        x0: initial guess of the opt. variables
        param_vec: multi-parametric input to opt. problem

        Outputs:
        sol: optimization output from IPOPT,
        (t1 - t0) * 1000: IPOPT solver runtime [ms]
        """

        b_warm = self.b_warm
        solver = self.solver
        lbx = self.lbx
        ubx = self.ubx
        lbg = self.lbg
        ubg = self.ubg
        if b_warm:
            lam_x0 = self.lam_x0
            lam_g0 = self.lam_g0

        ################################################################################################################
        # Solve problem
        ################################################################################################################
        t0 = time.perf_counter()
        if not lbx or not ubx:
            if b_warm:
                sol = solver(x0=x0, lbg=lbg, ubg=ubg, p=param_vec, lam_x0=lam_x0, lam_g0=lam_g0)
            else:
                sol = solver(x0=x0, lbg=lbg, ubg=ubg, p=param_vec)
        else:
            if b_warm:
                sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=param_vec, lam_x0=lam_x0, lam_g0=lam_g0)
            else:
                sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=param_vec)
        t1 = time.perf_counter()
        print("IPOPT time in ms: " + str((t1 - t0) * 1000) + " with status: " + solver.stats()['return_status'])

        return sol, (t1 - t0) * 1000

    def transform_sol(self,
                      sol: dict,
                      param_vec_: np.ndarray) -> tuple:
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.01.2020

        Documentation: Transforms IPOPT-solution back to state variables.

        Inputs:
        sol: IPOPT solution output
        param_vec: multi-parametric input into velocity optimization problem

        Outputs:
        v: optimized velocity [m/s]
        eps_tre: optimized tire slacks [-]
        F_p: optimize powertrain force [kN]
        """

        m = self.m
        trj = self.trj

        ################################################################################################################
        # Retrieve solution
        ################################################################################################################
        # --- Extract variables
        v = np.array(sol['x'][0:m])
        eps_tre = np.array(sol['x'][m:m + int(np.ceil(m / self.slack_every_v))])

        F_p, tire_cst, a, pwr_cst = trj(sol['x'], param_vec_)

        return v, eps_tre, np.array(F_p)


if __name__ == '__main__':
    pass
