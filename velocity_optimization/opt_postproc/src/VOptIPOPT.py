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
                 'vis_options',
                 'sol_dict',
                 'key',
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
                 'lam_g0',
                 'tire_model')

    def __init__(self,
                 m: int,
                 sid: str,
                 params_path: str,
                 vis_options: dict,
                 sol_dict: dict,
                 key: str,
                 b_warm: bool = False):
        """Class to construct the Interior-Point QP-optimizer IPOPT by defining the objective function f(x) \n
        and constraints g(x) to optimize a velocity profile for a given path.

        .. math::
            \mathrm{min} f(x) \n
            \mathrm{s.t.} \quad g(x) \leq 0

        More information to the IPOPT Solver can be found at https://coin-or.github.io/Ipopt/

        :param m: number of velocity points
        :param sid: optimized ID 'PerfSQP' or 'EmergSQP'
        :param slack_every_v: slack variable for every n velocity points
        :param params_path: absolute path to folder containing config file .ini
        :param vis_options: user specified visualization options of the debugging tool
        :param sol_options: user specified solver options of the debugging tool
        :param key: key of the used solver
        :param b_warm: allow or disallow the use of a warm start

        :Authors:
            Thomas Herrmann <thomas.herrmann@tum.de> \n
            Tobias Klotz <tobias.klotz@tum.de>

        :Created on:
            01.01.2020
        """

        self.m = m
        self.n = m - 1
        self.sid = sid
        self.unit_eps_tre = 0
        self.b_warm = b_warm
        self.vis_options = vis_options
        self.sol_dict = sol_dict
        self.key = key
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
        """Function to initialize the IPOPT solver by defining the objective function \n
        and constraints with the CasADi modeling language.

        :Authors:
            Thomas Herrmann <thomas.herrmann@tum.de> \n
            Tobias Klotz <tobias.klotz@tum.de>

        :Created on:
            01.01.2020
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

        # Slack Variable
        self.slack_every_v = opt_config.getint('SOLVER_GENERAL', 'slack_every_v')

        # Matrices
        one_vec = np.zeros((m, 1))
        one_vec[0] = 1  # Vector [1,0,...,0] mx1

        oneone_vec_short = np.ones((n, 1))  # Vector [1,...,1] nx1

        Mat_one_val = np.zeros((n, m))
        Mat_one_val[0, 0] = 1  # Matrix [1,0,...,0;0,...,0;0,...,0] nxm

        # Hard coded params --------------------------------------------------------------------------------------------
        # Vehicle parameters
        # max. power [kw] if constant max. power
        if self.sol_dict[self.key]['VarPower'] is False:
            # max. power of both e-machines [kW]
            P_max = opt_config.getfloat('VEHICLE', 'P_max_kW')
        # max. tractional force [kN]
        F_max = opt_config.getfloat('VEHICLE', 'F_max_kN')
        # max. braking force [kN]
        F_min = opt_config.getfloat('VEHICLE', 'F_min_kN')
        # vehicle mass [t]
        m_t = opt_config.getfloat('VEHICLE', 'mass_t')
        # max. veicle velocity [m/s]
        v_max = opt_config.getfloat('CAR_PARAMETER', 'v_max')  # [m/s] Maximum Velocity
        # max. acceleartion [m/s²]
        a_max = opt_config.getfloat('VEHICLE', 'ax_max_mps2')  # [m/s²] Maximum Laongitudianl Acceleration
        # max. lateral acceleration [m/s²]
        a_lat_max = opt_config.getfloat('VEHICLE', 'ay_max_mps2')  # [m/s²] Maximum Lateral Acceleration
        # max. slack angle [rad]
        beta_max = opt_config.getfloat('CAR_PARAMETER', 'beta_max')
        # max. gear rate [1/s]
        omega_max = opt_config.getfloat('CAR_PARAMETER', 'omega_max')
        # max. steer angle [rad]
        delta_max = opt_config.getfloat('CAR_PARAMETER', 'delta_max')
        # max. steer angle rate [rad/s]
        v_delta_max = opt_config.getfloat('CAR_PARAMETER', 'v_delta_max')
        # gravittational constatns [m/s²]
        grav = opt_config.getfloat('CAR_PARAMETER', 'grav')
        # rolling friction coefficient[-]
        f_r = opt_config.getfloat('CAR_PARAMETER', 'f_r')
        # length from CoG to Front Axle [m]
        l_f = opt_config.getfloat('CAR_PARAMETER', 'l_f')
        # length from CoG to Rear Axle [m]
        l_r = opt_config.getfloat('CAR_PARAMETER', 'l_r')
        # Front Surface [m²]
        A = opt_config.getfloat('CAR_PARAMETER', 'A')
        # Air Density [kg/m³]
        rho = opt_config.getfloat('CAR_PARAMETER', 'rho')
        # Distribution of Engine Force (Front/Rear) [-]
        k_dr = opt_config.getfloat('CAR_PARAMETER', 'k_dr')
        # Distribution of Braking Force (Front/Rear) [-]
        k_br = opt_config.getfloat('CAR_PARAMETER', 'k_br')
        # Distribution of lateral Tire Force
        k_roll = opt_config.getfloat('CAR_PARAMETER', 'k_roll')
        # Drift Coefficient at Front Tire [-]
        c_lf = opt_config.getfloat('CAR_PARAMETER', 'c_lf')
        # Drift Coefficient at Rear Tire [-]
        c_lr = opt_config.getfloat('CAR_PARAMETER', 'c_lr')
        # Height of CoG [m]
        h_cg = opt_config.getfloat('CAR_PARAMETER', 'h_cg')
        # Yaw Inertia Coefficient [t m²]
        I_zz = opt_config.getfloat('CAR_PARAMETER', 'I_zz')
        # Spurbreite vorne
        tw_f = opt_config.getfloat('CAR_PARAMETER', 'tw_f')
        # Spurbreite hinten
        tw_r = opt_config.getfloat('CAR_PARAMETER', 'tw_r')
        # Tire Model (Extended Magic Formula of Fabian Christ)
        F_z0 = opt_config.getfloat('CAR_PARAMETER', 'F_z0')
        B_f = opt_config.getfloat('CAR_PARAMETER', 'B_f')
        C_f = opt_config.getfloat('CAR_PARAMETER', 'C_f')
        D_f = opt_config.getfloat('CAR_PARAMETER', 'D_f')
        E_f = opt_config.getfloat('CAR_PARAMETER', 'E_f')
        eps_f = opt_config.getfloat('CAR_PARAMETER', 'eps_f')
        B_r = opt_config.getfloat('CAR_PARAMETER', 'B_r')
        C_r = opt_config.getfloat('CAR_PARAMETER', 'C_r')
        D_r = opt_config.getfloat('CAR_PARAMETER', 'D_r')
        E_r = opt_config.getfloat('CAR_PARAMETER', 'E_r')
        eps_r = opt_config.getfloat('CAR_PARAMETER', 'eps_r')

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
        penalty_jerk = opt_config.getfloat(set_sec, 'penalty_jerk')

        # --- Variable friction
        if self.sol_dict[self.key]['VarFriction']:
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
        # PM -----------------------------------------------------------------------------------------------------------
        ################################################################################################################
        if self.sol_dict[self.key]['Model'] == "PM":

            ############################################################################################################
            # Optimization Variables
            ############################################################################################################
            # Velocity [m/s]
            v = cs.SX.sym('v', m)

            # Tire slack variables epsilon [-]
            if self.sol_dict[self.key]['Slack']:
                eps_tre = cs.SX.sym('eps_tre', int(np.ceil(m / self.slack_every_v)))

            ############################################################################################################
            # Online Parameters
            ############################################################################################################
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

            # max. power if variable power
            if self.sol_dict[self.key]['VarPower']:
                # max. power [kW]
                P_max_param = cs.SX.sym('P_max_param', m - 1)

            # max. acceleration in x- and y-direction of the vehicle if variable power
            if ax_max is None:
                ax_max = cs.SX.sym('ax_max', m - 1)
                ay_max = cs.SX.sym('ay_max', m - 1)

            ############################################################################################################
            # Constraints
            ############################################################################################################
            # acceleration [m/s^2]
            acc = (v[1:] ** 2 - v[: - 1] ** 2) / (2 * delta_s_param)

            # --- Powertrain force [kN]
            F_p = m_t * acc + c_res * (v ** 2)[0: m - 1] * 0.001

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
            if self.sol_dict[self.key]['VarPower'] is False:
                pwr_cst = F_p * v[:-1]
                g.append(pwr_cst)
                ubg.append([P_max] * n)
                lbg.append([-np.inf] * n)
            else:
                pwr_cst = F_p * v[:-1] - P_max_param[0:]
                g.append(pwr_cst)
                ubg.append([0] * n)
                lbg.append([-np.inf] * n)

            # --- Tires

            # Friction Diamond
            if self.sol_dict[self.key]['Friction'] == "Diamond":
                # (| a_x_mps2 | __i / ax_max + a | a_y_mps2 | __i / ay_max - 1__i - eps <= 0__i)
                # tire_cst = 1.0 * \
                #    (1 / ax_max * 1 / m_t * F_p) ** 2 + 1.0 * (1 / ay_max * mtimes(d, (v ** 2))) ** 2
                tire_cst1 = (F_p / m_t) / ax_max + (kappa_param * (v ** 2))[: - 1] / ay_max - oneone_vec_short
                tire_cst2 = (F_p / m_t) / ax_max - (kappa_param * (v ** 2))[: - 1] / ay_max - oneone_vec_short
                tire_cst3 = - (F_p / m_t) / ax_max - (kappa_param * (v ** 2))[: - 1] / ay_max - oneone_vec_short
                tire_cst4 = - (F_p / m_t) / ax_max + (kappa_param * (v ** 2))[: - 1] / ay_max - oneone_vec_short

                # Slack variables used
                if self.sol_dict[self.key]['Slack']:
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

            # Kamm circle/ellipse
            elif self.sol_dict[self.key]['Friction'] == "Circle":
                if ax_max is None:
                    # constant acceleration limit
                    tire_cst1 = (kappa_param[: - 1] * v[: - 1] ** 2) ** 2 + acc ** 2
                    g.append(tire_cst1)
                    ubg.append([a_lat_max ** 2] * n)
                    lbg.append([- np.inf] * n)
                else:
                    # variable acceleration limit
                    tire_cst1 = acc ** 2 / ax_max ** 2 + (kappa_param[: - 1]
                                                          * (v[: - 1] ** 2))**2 / ay_max**2 - oneone_vec_short
                    g.append(tire_cst1)
                    ubg.append([0] * n)
                    lbg.append([- np.inf] * n)

                # Slack variables used
                if self.sol_dict[self.key]['Slack']:
                    i = 0
                    # counter variable for slacks
                    j = 0
                    for tre in range(tire_cst1.shape[0]):
                        # if i reaches switching point, go to next slack variable and apply to following tire entries
                        if np.mod(i, self.slack_every_v) == 0 and i >= self.slack_every_v:
                            j += 1
                        tire_cst1[i] -= unit_eps_tre * eps_tre[j]
                        i += 1

            # --- Initial value velocity
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

            # --- Slack variables
            if self.sol_dict[self.key]['Slack']:
                # inf >= [eps_tre__1,...,eps_tre__m] >= 0
                eps_tre_cst = unit_eps_tre * eps_tre
                g.append(eps_tre_cst)
                lbg.append([0] * int(np.ceil(m / self.slack_every_v)))
                ubg.append([unit_eps_tre * eps_tre_max] * int(np.ceil(m / self.slack_every_v)))

            ############################################################################################################
            # Objective function
            ############################################################################################################
            # sum__i (d_s_m__i * 1/v__i + s_eps_vel * eps_vel_mps__i + s_eps_dF * (dF__i)**2 + s_eps_tre * eps_tre__i)
            if self.sol_dict[self.key]['Slack']:
                J = cs.sum1((v - v_max_param) ** 2) + \
                    penalty_jerk * cs.sum1((cs.diff(cs.diff(v))) ** 2) + \
                    s_eps_tre_lin * cs.sum1(unit_eps_tre * eps_tre) + \
                    s_eps_tre_quad * cs.sum1((unit_eps_tre * eps_tre) ** 2)
            else:
                J = cs.sum1((v - v_max_param) ** 2) + \
                    penalty_jerk * cs.sum1((cs.diff(cs.diff(v))) ** 2)

            ############################################################################################################
            # Solver stuff
            ############################################################################################################
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
                if self.sol_dict[self.key]['VarPower'] is False:
                    param_vec = cs.vertcat(v_ini_param,
                                           kappa_param,
                                           delta_s_param,
                                           v_max_param,
                                           F_cst_ini_param,
                                           v_end_param)
                else:
                    param_vec = cs.vertcat(v_ini_param,
                                           kappa_param,
                                           delta_s_param,
                                           v_max_param,
                                           F_cst_ini_param,
                                           v_end_param,
                                           P_max_param)
            # Variable friction
            else:
                if self.sol_dict[self.key]['VarPower'] is False:
                    param_vec = cs.vertcat(v_ini_param,
                                           kappa_param,
                                           delta_s_param,
                                           v_max_param,
                                           F_cst_ini_param,
                                           v_end_param,
                                           ax_max,
                                           ay_max)
                else:
                    param_vec = cs.vertcat(v_ini_param,
                                           kappa_param,
                                           delta_s_param,
                                           v_max_param,
                                           F_cst_ini_param,
                                           v_end_param,
                                           P_max_param,
                                           ax_max,
                                           ay_max)

            # Formatting optimization vector for optimization
            if self.sol_dict[self.key]['Slack']:
                x = cs.vertcat(v, eps_tre)  # create optimization variables vector
            else:
                x = cs.vertcat(v)   # create optimization variables vector

            # create nonlinear problem
            nlp = {'x': x, 'p': param_vec, 'f': J, 'g': g}

            # solver options
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

        ################################################################################################################
        # KM -----------------------------------------------------------------------------------------------------------
        ################################################################################################################

        elif self.sol_dict[self.key]['Model'] == "KM":
            ############################################################################################################
            # Optimization Variables
            ############################################################################################################
            # Velocity [m/s]
            v = cs.SX.sym('v', m)

            # Tire slack variables epsilon [-]
            if self.sol_dict[self.key]['Slack']:
                eps_tre = cs.SX.sym('eps_tre', int(np.ceil(m / self.slack_every_v)))

            ############################################################################################################
            # Online Parameters
            ############################################################################################################
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

            # max. power if variable power
            if self.sol_dict[self.key]['VarPower']:
                # max. power [kW]
                P_max_param = cs.SX.sym('P_max_param', m - 1)

            # max. acceleration if variable acceleration
            if ax_max is None:
                ax_max = cs.SX.sym('ax_max', m - 1)
                ay_max = cs.SX.sym('ay_max', m - 1)

            ############################################################################################################
            # Constraints
            ############################################################################################################
            # acceleration [m/s^2]
            acc = (v[1:] ** 2 - v[: - 1] ** 2) / (2 * delta_s_param)

            # --- Powertrain force [kN]
            F_p = m_t * acc + c_res * (v ** 2)[0: m - 1] * 0.001

            # Steer Angle
            delta = cs.atan2(kappa_param * (l_f + l_r), 1)

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

            # --- Steer Angle Rate
            delta_ini = (delta[1:self.m] - delta[0:self.m - 1]) / (delta_s_param / v[0:self.m - 1])
            g.append(delta_ini)
            ubg.append([v_delta_max] * n)
            lbg.append([-v_delta_max] * n)

            # --- Power
            # P_max * [v_mps__1, ... v_mps__n] >= [F_p__1, ..., F_p__n]
            # 0 >= F_p__i - P_max * 1/v_mps__i >= - inf
            if self.sol_dict[self.key]['VarPower'] is False:
                pwr_cst = F_p * v[:-1]
                g.append(pwr_cst)
                ubg.append([P_max] * n)
                lbg.append([-np.inf] * n)
            else:
                pwr_cst = F_p * v[:-1] - P_max_param[0:]
                g.append(pwr_cst)
                ubg.append([0] * n)
                lbg.append([-np.inf] * n)

            # --- Tires

            if self.sol_dict[self.key]['Friction'] == "Diamond":
                # (| a_x_mps2 | __i / ax_max + a | a_y_mps2 | __i / ay_max - 1__i - eps <= 0__i)
                # tire_cst = 1.0 * \
                #    (1 / ax_max * 1 / m_t * F_p) ** 2 + 1.0 * (1 / ay_max * mtimes(d, (v ** 2))) ** 2
                tire_cst1 = (F_p / m_t) / ax_max + (kappa_param * (v ** 2))[: - 1] / ay_max - oneone_vec_short
                tire_cst2 = (F_p / m_t) / ax_max - (kappa_param * (v ** 2))[: - 1] / ay_max - oneone_vec_short
                tire_cst3 = - (F_p / m_t) / ax_max - (kappa_param * (v ** 2))[: - 1] / ay_max - oneone_vec_short
                tire_cst4 = - (F_p / m_t) / ax_max + (kappa_param * (v ** 2))[: - 1] / ay_max - oneone_vec_short

                if self.sol_dict[self.key]['Slack']:
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

            elif self.sol_dict[self.key]['Friction'] == "Circle":

                if self.sol_dict[self.key]['VarFriction']:
                    # variable acceleration limit
                    tire_cst1 = acc ** 2 / ax_max ** 2 + (kappa_param[: - 1]
                                                          * (v[: - 1] ** 2)) ** 2 / ay_max ** 2 - oneone_vec_short
                    # Slack variables used
                    if self.sol_dict[self.key]['Slack']:
                        i = 0
                        # counter variable for slacks
                        j = 0
                        for tre in range(tire_cst1.shape[0]):
                            # if i reaches switching point, go to next slack variable and apply to following tire
                            # entries
                            if np.mod(i, self.slack_every_v) == 0 and i >= self.slack_every_v:
                                j += 1
                            tire_cst1[i] -= unit_eps_tre * eps_tre[j]
                            i += 1

                    g.append(tire_cst1)
                    ubg.append([0] * n)
                    lbg.append([- np.inf] * n)

                else:
                    # constant acceleration limit
                    tire_cst1 = (kappa_param[: - 1] * v[: - 1] ** 2) ** 2 + acc ** 2
                    # Slack variables used
                    if self.sol_dict[self.key]['Slack']:
                        i = 0
                        # counter variable for slacks
                        j = 0
                        for tre in range(tire_cst1.shape[0]):
                            # if i reaches switching point, go to next slack variable and apply to following tire
                            # entries
                            if np.mod(i, self.slack_every_v) == 0 and i >= self.slack_every_v:
                                j += 1
                            tire_cst1[i] -= unit_eps_tre * eps_tre[j]
                            i += 1

                    g.append(tire_cst1)
                    ubg.append([a_lat_max ** 2] * n)
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

            ############################################################################################################
            # Objective function
            ############################################################################################################
            # sum__i (d_s_m__i * 1/v__i + s_eps_vel * eps_vel_mps__i + s_eps_dF * (dF__i)**2 + s_eps_tre * eps_tre__i)
            if self.sol_dict[self.key]['Slack']:
                J = cs.sum1((v - v_max_param) ** 2) + \
                    penalty_jerk * cs.sum1((cs.diff(cs.diff(v))) ** 2) + \
                    s_eps_tre_lin * cs.sum1(unit_eps_tre * eps_tre) + \
                    s_eps_tre_quad * cs.sum1((unit_eps_tre * eps_tre) ** 2)
            else:
                J = cs.sum1((v - v_max_param) ** 2)

            ############################################################################################################
            # Solver stuff
            ############################################################################################################
            # Initialization of optimization variables, OVERRIDDEN during usage
            x0.append([20] * m)  # initial guess for velocity

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
            if self.sol_dict[self.key]['VarFriction']:
                if self.sol_dict[self.key]['VarPower'] is False:
                    param_vec = cs.vertcat(v_ini_param,
                                           kappa_param,
                                           delta_s_param,
                                           v_max_param,
                                           F_cst_ini_param,
                                           v_end_param,
                                           ax_max,
                                           ay_max)
                else:
                    param_vec = cs.vertcat(v_ini_param,
                                           kappa_param,
                                           delta_s_param,
                                           v_max_param,
                                           F_cst_ini_param,
                                           v_end_param,
                                           P_max_param,
                                           ax_max,
                                           ay_max)
            # Variable friction
            else:
                if self.sol_dict[self.key]['VarPower'] is False:
                    param_vec = cs.vertcat(v_ini_param,
                                           kappa_param,
                                           delta_s_param,
                                           v_max_param,
                                           F_cst_ini_param,
                                           v_end_param)
                else:
                    param_vec = cs.vertcat(v_ini_param,
                                           kappa_param,
                                           delta_s_param,
                                           v_max_param,
                                           F_cst_ini_param,
                                           v_end_param,
                                           P_max_param)

            # Formatting optimization vector for optimization
            if self.sol_dict[self.key]['Slack']:
                x = cs.vertcat(v, eps_tre)  # create optimization variables vector
            else:
                x = cs.vertcat(v)  # create optimization variables vector

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

        ################################################################################################################
        # DM -----------------------------------------------------------------------------------------------------------
        ################################################################################################################

        elif self.sol_dict[self.key]['Model'] == "DM":
            ############################################################################################################
            # Optimization Variables
            ############################################################################################################
            # velocity [m/s]
            v = cs.SX.sym('v', m)

            # slack angle [rad]
            beta = cs.SX.sym('beta', m)

            # gear rate [1/s]
            omega = cs.SX.sym('omega', m)

            # driving force [kN]
            F_dr = cs.SX.sym('F_dr', n)

            # brake force [kN]
            F_br = cs.SX.sym('F_br', n)

            # steer angle [rad]
            delta = cs.SX.sym('delta', n)

            ############################################################################################################
            # Online Parameters
            ############################################################################################################
            # step length [m]
            ds = cs.SX.sym('ds', n)

            # curvature [1/m]
            kappa = cs.SX.sym('kappa', m)

            # max. velocity [m/s]
            v_max_param = cs.SX.sym('v_max', m)

            # initial velocity [m/s]
            v_ini_param = cs.SX.sym('v_ini', 1)

            # end velocity [m/s]
            v_end_param = cs.SX.sym('v_end', 1)

            # max. power if variable power
            if self.sol_dict[self.key]['VarPower']:
                # max. power [kW]
                P_max_param = cs.SX.sym('P_max_param', m - 1)

            # max acceleration in x-direction [m/s²]
            ax_max = cs.SX.sym('ax_max', n)

            # max accelearation in y-direction [m/s²]
            ay_max = cs.SX.sym('ay_max', n)

            ############################################################################################################
            # Objective function
            ############################################################################################################
            # sum__i (d_s_m__i * 1/v__i)
            J = cs.sum1(ds / v[1::])

            ############################################################################################################
            # Boundary conditions
            ############################################################################################################
            # --- BOUNDARY CONDITIONS
            # --- Velocity

            # --- Start value velocity
            # v_0 = v_ini
            v_cst_0 = v[0] - v_ini_param
            g.append(v_cst_0)
            ubg.append([0.0])
            lbg.append([0.0])

            # --- v_min <= v_1, ... v_end_1 <= v_max
            v_cst = v[1:-1]
            g.append(v_cst)
            ubg.append([v_max] * (m - 2))
            lbg.append([0.0] * (m - 2))

            # --- End value velocity
            # v_mps__N-1 == v_end_param
            g.append(v[-1] - v_end_param)
            ubg.append([0.0])
            lbg.append([- np.inf])

            # --- Slip angle
            # - beta_max <= [beta__0, ...,beta__N] <= beta_max
            beta_cst = beta
            g.append(beta_cst)
            ubg.append([beta_max] * m)
            lbg.append([- beta_max] * m)

            # --- Gear rate
            # - omega_max <= [omega__0, ..., omega__N] <= omega_max
            omega_cst = omega
            g.append(omega_cst)
            ubg.append([omega_max] * m)
            lbg.append([- omega_max] * m)

            # --- Drive force
            # 0 <= [F_dr__0, ..., F_dr__N-1] <= F_dr_max
            F_dr_cst = F_dr
            g.append(F_dr_cst)
            ubg.append([F_max] * n)
            lbg.append([0.0] * n)

            # --- Brake force
            # F_br_max <= [F_br__0, ..., F_br__N-1] <= 0.0
            F_br_cst = F_br
            g.append(F_br_cst)
            ubg.append([0.0] * n)
            lbg.append([F_min] * n)

            # --- Steer angle
            # -delta_max <= [delta__1, ..., delta__N] <= delta_max
            delta_cst = delta
            g.append(delta_cst)
            ubg.append([delta_max] * n)
            lbg.append([- delta_max] * n)

            ############################################################################################################
            # Forces
            ############################################################################################################
            # roll friction (front & rear)
            F_roll_f = f_r * m_t * grav * l_r / (l_r + l_f)
            F_roll_r = f_r * m_t * grav * l_f / (l_r + l_f)

            # tire slip angle (front & rear)
            alpha_f = delta - cs.atan2((l_f * kappa[: - 1] * v[: - 1] / (2 * np.pi) + v[: - 1] * cs.sin(beta[: - 1])),
                                       v[: - 1] * cs.cos(beta[: - 1]))
            alpha_r = cs.atan2(l_r * kappa[: - 1] * v[: - 1] / (2 * np.pi) - v[: - 1] * cs.sin(beta[: - 1]),
                               v[: - 1] * cs.cos(beta[: - 1]))

            # air resistance
            F_d = 0.001 * c_res * v[: - 1] ** 2

            # force at axle in x-direction (front & rear)
            F_xf = k_dr * F_dr + k_br * F_br - F_roll_f
            F_xr = (1 - k_dr) * F_dr + (1 - k_br) * F_br - F_roll_r

            # total force in x-direction at CoG
            ma_x = F_xf + F_xr - F_d

            # normal force at axle (front & rear)
            F_zf = m_t * grav * l_r / (l_r + l_f) - h_cg / (l_r + l_f) * ma_x + 0.5 * c_lf * rho * A * v[: - 1] ** 2
            F_zr = m_t * grav * l_f / (l_r + l_f) + h_cg / (l_r + l_f) * ma_x + 0.5 * c_lr * rho * A * v[: - 1] ** 2

            # force in y-direction at axle (front & rear)
            F_yf = D_f * (1 + eps_f * F_zf / F_z0) * F_zf / F_z0 * cs.sin(
                C_f * cs.atan2(B_f * alpha_f - E_f * (B_f * alpha_f - cs.atan2(B_f * alpha_f, 1)), 1))
            F_yr = D_r * (1 + eps_r * F_zr / F_z0) * F_zr / F_z0 * cs.sin(
                C_r * cs.atan2(B_r * alpha_r - E_r * (B_r * alpha_r - cs.atan2(B_r * alpha_r, 1)), 1))

            # total lateral accelaration at CoG
            ma_y = F_yr + F_xf * cs.sin(delta) + F_yf * cs.cos(delta)

            ############################################################################################################
            # Equality Constraints
            ############################################################################################################
            # --- dot_v = ....
            v_dot_cst = v[1:] - v[: - 1] - ds \
                / v[: - 1] * (1 / m_t
                              * (+ F_xr * cs.cos(beta[: - 1])
                                 + F_xf * cs.cos(delta - beta[: - 1])
                                 + F_yr * cs.sin(beta[: - 1]) - F_yf
                                 * cs.sin(delta - beta[: - 1]) - F_d * cs.cos(beta[: - 1])))
            g.append(v_dot_cst)
            ubg.append([0.0] * n)
            lbg.append([0.0] * n)

            # --- dot_beta = ...
            beta_dot_cst = (beta[1:] - beta[:-1]) / (ds / v[:-1]) \
                - 1 / (2 * np.pi) * (-kappa[: - 1] * v[: - 1] + 1 / (m_t * v[:-1])
                                     * (- F_xr * cs.sin(beta[:-1]) + F_xf * cs.sin(delta - beta[:-1])
                                        + F_yr * cs.cos(beta[:-1]) + F_yf * cs.cos(delta - beta[:-1])
                                        + F_d * cs.sin(beta[:-1])))
            g.append(beta_dot_cst)
            ubg.append([0.0] * n)
            lbg.append([0.0] * n)

            # --- dot_omega = ...
            omega_dot_cst = kappa[1:] * v[1:] - kappa[1:] * v[:-1] \
                - ds / v[: - 1] * (1 / I_zz * (F_xf * cs.sin(delta) * l_f + F_yf * cs.cos(delta) * l_f - F_yr * l_r))
            g.append(omega_dot_cst)
            ubg.append([0.0] * n)
            lbg.append([0.0] * n)

            ############################################################################################################
            # Inequality Constraints
            ############################################################################################################

            # --- 0 <= v * F_dr <= P_max
            if self.sol_dict[self.key]['VarPower'] is False:
                pwr_cst = F_dr * v[:-1]
                g.append(pwr_cst)
                ubg.append([P_max] * n)
                lbg.append([-np.inf] * n)
            else:
                pwr_cst = F_dr * v[:-1] - P_max_param[0:]
                g.append(pwr_cst)
                ubg.append([0] * n)
                lbg.append([-np.inf] * n)

            # --- -0.02 <= F_br * F_dr <= 0
            F_dr_F_br_cst = F_br * F_dr
            g.append(F_dr_F_br_cst)
            ubg.append([0.0] * n)
            lbg.append([-0.02] * n)

            # --- Friction circle
            mu_x = ax_max / a_max
            mu_y = ay_max / a_lat_max

            # Front tire
            fric_front_tire_cst = (F_xf / (mu_x * F_zf)) ** 2 + (F_yf / (mu_y * F_zf)) ** 2
            g.append(fric_front_tire_cst)
            ubg.append([1.0] * n)
            lbg.append([0.0] * n)

            # Rear tire
            fric_rear_tire_cst = (F_xr / (mu_x * F_zr)) ** 2 + (F_yr / (mu_y * F_zr)) ** 2
            g.append(fric_rear_tire_cst)
            ubg.append([1.0] * n)
            lbg.append([0.0] * n)

            # Change Rate of Driving Force
            dot_f_dr_cst = (F_dr[1:] - F_dr[:-1]) / (ds[:-1] / v[:-2])
            g.append(dot_f_dr_cst)
            ubg.append([F_max] * (m - 2))
            lbg.append([-np.inf] * (m - 2))

            # Change Rate of Braking Force
            dot_f_br_cst = (F_br[1:] - F_br[:-1]) / (ds[:-1] / v[:-2])
            g.append(dot_f_br_cst)
            ubg.append([np.inf] * (m - 2))
            lbg.append([F_min] * (m - 2))

            # Change Rate of Steering Angle
            dot_delta_cst = (delta[1:] - delta[:-1]) / (ds[:-1] / v[:-2])
            g.append(dot_delta_cst)
            ubg.append([delta_max] * (m - 2))
            lbg.append([- delta_max] * (m - 2))

            ############################################################################################################
            # Solver stuff
            ############################################################################################################
            # Initialization of optimization variables, OVERRIDDEN during usage
            x0.append([10] * m)  # initial guess for velocity
            x0.append([0] * m)  # initial guess for slip angle
            x0.append([0] * m)  # initial guess for gear rate
            x0.append([0] * n)  # initial guess for driving force
            x0.append([0] * n)  # initial guess for braking force
            x0.append([0] * n)  # initial guess for steering rate

            x0 = np.concatenate(x0)

            # Formatting lower and upper bounds of optimization variables if available

            # Formatting constraints
            g = cs.vertcat(*g)
            lbg = cs.vertcat(*lbg)
            ubg = cs.vertcat(*ubg)

            # --- Formatting parameter vector for optimization
            # Constant friction
            if self.sol_dict[self.key]['VarPower'] is False:
                param_vec = cs.vertcat(kappa,
                                       ds,
                                       v_ini_param,
                                       v_end_param,
                                       ax_max,
                                       ay_max)
            else:
                param_vec = cs.vertcat(kappa,
                                       ds,
                                       v_ini_param,
                                       v_end_param,
                                       P_max_param,
                                       ax_max,
                                       ay_max)

            # Formatting optimization vector for optimization
            x = cs.vertcat(v, beta, omega, F_dr, F_br, delta)  # create optimization variables vector

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

            self.trj = []
            self.x = x
            self.x0 = x0
            self.lbx = lbx
            self.ubx = ubx
            self.g = g
            self.lbg = lbg
            self.ubg = ubg
            self.J = 0

        ################################################################################################################
        # FW -----------------------------------------------------------------------------------------------------------
        ################################################################################################################

        elif self.sol_dict[self.key]['Model'] == "FW":
            ############################################################################################################
            # Optimization Variables
            ############################################################################################################
            # velocity [m/s]
            v = cs.SX.sym('v', m)

            # slack angle [rad]
            beta = cs.SX.sym('beta', m)

            # driving force [kN]
            F_dr = cs.SX.sym('F_dr', n)

            # brake force [kN]
            F_br = cs.SX.sym('F_br', n)

            # steer angle [rad]
            delta = cs.SX.sym('delta', n)

            # steer angle [rad]
            gamma_y = cs.SX.sym('gamma', n)

            ############################################################################################################
            # Online Parameters
            ############################################################################################################
            # step length [m]
            ds = cs.SX.sym('ds', n)

            # curvature [1/m]
            kappa = cs.SX.sym('kappa', m)

            # max. velocity [m/s]
            v_max_param = cs.SX.sym('v_max', m)

            # initial velocity [m/s]
            v_ini_param = cs.SX.sym('v_ini', 1)

            # end velocity [m/s]
            v_end_param = cs.SX.sym('v_end', 1)

            # max. power if variable power
            if self.sol_dict[self.key]['VarPower']:
                # max. power [kW]
                P_max_param = cs.SX.sym('P_max_param', m - 1)

            # max. acceleration in x- and y-direction of the vehicle if variable power
            # max acceleration in x-direction [m/s²]
            ax_max = cs.SX.sym('ax_max', n)

            # max accelearation in y-direction [m/s²]
            ay_max = cs.SX.sym('ay_max', n)

            ############################################################################################################
            # Objective function
            ############################################################################################################
            # sum__i (d_s_m__i * 1/v__i)
            J = cs.sum1(ds / v[1::])

            ############################################################################################################
            # Boundary conditions
            ############################################################################################################
            # --- BOUNDARY CONDITIONS
            # --- Velocity

            # --- Start value velocity
            # v_0 = v_ini
            v_cst_0 = v[0] - v_ini_param
            g.append(v_cst_0)
            ubg.append([0.0])
            lbg.append([0.0])

            # --- v_min <= v_1, ... v_end_1 <= v_max
            v_cst = v[1:-1]
            g.append(v_cst)
            ubg.append([v_max] * (m - 2))
            lbg.append([0.0] * (m - 2))

            # --- End value velocity
            # v_mps__N-1 == v_end_param
            g.append(v[-1] - v_end_param)
            ubg.append([0.0])
            lbg.append([- np.inf])

            # --- Slip angle
            # - beta_max <= [beta__0, ...,beta__N] <= beta_max
            beta_cst = beta
            g.append(beta_cst)
            ubg.append([beta_max] * m)
            lbg.append([- beta_max] * m)

            # --- Drive force
            # 0 <= [F_dr__0, ..., F_dr__N-1] <= F_dr_max
            F_dr_cst = F_dr
            g.append(F_dr_cst)
            ubg.append([F_max] * n)
            lbg.append([0.0] * n)

            # --- Breake force
            # F_br_max <= [F_br__0, ..., F_br__N-1] <= 0.0
            F_br_cst = F_br
            g.append(F_br_cst)
            ubg.append([0.0] * n)
            lbg.append([F_min] * n)

            # --- Steer angle
            # -delta_max <= [delta__1, ..., delta__N] <= delta_max
            delta_cst = delta
            g.append(delta_cst)
            ubg.append([delta_max] * n)
            lbg.append([- delta_max] * n)

            # --- Lateral Tire Force Distribution
            # -gamma_max <= [gamma__1, ..., gamma__N] <= gamma_max
            gamma_cst = gamma_y
            g.append(gamma_cst)
            ubg.append([np.inf] * n)
            lbg.append([- np.inf] * n)

            ############################################################################################################
            # Forces
            ############################################################################################################
            # roll friction (front & rear/ left & right) [kN]
            F_roll_fl = 0.5 * f_r * m_t * grav * l_r / (l_r + l_f)
            F_roll_fr = 0.5 * f_r * m_t * grav * l_r / (l_r + l_f)
            F_roll_rl = 0.5 * f_r * m_t * grav * l_f / (l_r + l_f)
            F_roll_rr = 0.5 * f_r * m_t * grav * l_f / (l_r + l_f)

            # tire slip angle (front & rear/ left & right) [rad]
            alpha_fl = delta - cs.atan2((l_f * kappa[: - 1] * v[: - 1] / (2 * np.pi) + v[: - 1] * cs.sin(beta[: - 1])),
                                        v[: - 1] * cs.cos(beta[: - 1]) - 0.5 * tw_f * kappa[: - 1] * v[: - 1]
                                        / (2 * np.pi))
            alpha_fr = delta - cs.atan2((l_f * kappa[: - 1] * v[: - 1] / (2 * np.pi) + v[: - 1] * cs.sin(beta[: - 1])),
                                        v[: - 1] * cs.cos(beta[: - 1]) + 0.5 * tw_f * kappa[: - 1] * v[: - 1]
                                        / (2 * np.pi))

            alpha_rl = cs.atan2(l_r * kappa[: - 1] * v[: - 1] / (2 * np.pi) - v[: - 1] * cs.sin(beta[: - 1]),
                                v[: - 1] * cs.cos(beta[: - 1]) - 0.5 * tw_r * kappa[: - 1] * v[: - 1] / (2 * np.pi))
            alpha_rr = cs.atan2(l_r * kappa[: - 1] * v[: - 1] / (2 * np.pi) - v[: - 1] * cs.sin(beta[: - 1]),
                                v[: - 1] * cs.cos(beta[: - 1]) + 0.5 * tw_r * kappa[: - 1] * v[: - 1] / (2 * np.pi))

            # air resistance [kN]
            F_d = 0.001 * c_res * v[: - 1] ** 2

            # tire force in x-direction (front & rear/ left & right) [kN]
            F_xfl = 0.5 * k_dr * F_dr + 0.5 * k_br * F_br - F_roll_fl
            F_xfr = 0.5 * k_dr * F_dr + 0.5 * k_br * F_br - F_roll_fr

            F_xrl = 0.5 * (1 - k_dr) * F_dr + 0.5 * (1 - k_br) * F_br - F_roll_rl
            F_xrr = 0.5 * (1 - k_dr) * F_dr + 0.5 * (1 - k_br) * F_br - F_roll_rr

            # total force in x-direction at CoG [kN]
            ma_x = F_xrl + F_xrr + (F_xfl + F_xfr) - F_d

            # tire normal force (front & rear / left & right) [kN]
            F_zfl = + 0.5 * m_t * grav * l_r / (l_r + l_f) - 0.5 * h_cg / (l_r + l_f) * ma_x - k_roll * gamma_y \
                    + 0.25 * c_lf * rho * A * v[: - 1] ** 2

            F_zfr = + 0.5 * m_t * grav * l_r / (l_r + l_f) - 0.5 * h_cg / (l_r + l_f) * ma_x \
                    + k_roll * gamma_y \
                    + 0.25 * c_lf * rho * A * v[: - 1] ** 2

            F_zrl = + 0.5 * m_t * grav * l_f / (l_r + l_f) \
                    + 0.5 * h_cg / (l_r + l_f) * ma_x - (1 - k_roll) * gamma_y \
                    + 0.25 * c_lr * rho * A * v[: - 1] ** 2

            F_zrr = + 0.5 * m_t * grav * l_f / (l_r + l_f) \
                    + 0.5 * h_cg / (l_r + l_f) * ma_x \
                    + (1 - k_roll) * gamma_y \
                    + 0.25 * c_lr * rho * A * v[: - 1] ** 2

            # tire force in y-direction (front & rear/ left & right) [kN]
            F_yfl = D_f * (1 + eps_f * F_zfl / F_z0) * F_zfl / F_z0 \
                * cs.sin(C_f * cs.atan2(B_f * alpha_fl - E_f * (B_f * alpha_fl - cs.atan2(B_f * alpha_fl, 1)), 1))
            F_yfr = D_f * (1 + eps_f * F_zfr / F_z0) * F_zfr / F_z0 \
                * cs.sin(C_f * cs.atan2(B_f * alpha_fr - E_f * (B_f * alpha_fr - cs.atan2(B_f * alpha_fr, 1)), 1))

            F_yrl = D_r * (1 + eps_r * F_zrl / F_z0) * F_zrl / F_z0 \
                * cs.sin(C_r * cs.atan2(B_r * alpha_rl - E_r * (B_r * alpha_rl - cs.atan2(B_r * alpha_rl, 1)), 1))
            F_yrr = D_r * (1 + eps_r * F_zrr / F_z0) * F_zrr / F_z0 \
                * cs.sin(C_r * cs.atan2(B_r * alpha_rr - E_r * (B_r * alpha_rr - cs.atan2(B_r * alpha_rr, 1)), 1))

            # total force in y-direction at CoG
            ma_y = F_yrl + F_yrr + (F_xfl + F_xfr) * cs.sin(delta) + (F_yfl + F_yfr) * cs.cos(delta)

            ############################################################################################################
            # Equality Constraints
            ############################################################################################################
            # --- dot_v = ....
            v_dot_cst = v[1:] - v[: - 1] - ds \
                / v[: - 1] * (1 / m_t * (+ (F_xrl + F_xrr) * cs.cos(beta[:-1])
                                         + (F_xfl + F_xfr) * cs.cos(delta - beta[:-1])
                                         + (F_yrl + F_yrr) * cs.sin(beta[:-1])
                                         - (F_yfl + F_yfr) * cs.sin(delta - beta[:-1])
                                         - F_d * cs.cos(beta[:-1])))
            g.append(v_dot_cst)
            ubg.append([0.0] * n)
            lbg.append([0.0] * n)

            # --- dot_beta = ...
            beta_dot_cst = (beta[1:] - beta[:-1]) \
                / (ds / v[:-1]) - 1 / (2 * np.pi) * (-kappa[: - 1] * v[: - 1]
                                                     + 1 / (m_t * v[:-1])
                                                     * (- (F_xrl + F_xrr) * cs.sin(beta[:-1])
                                                        + (F_xfl + F_xfr) * cs.sin(delta - beta[:-1])
                                                        + (F_yrl + F_yrr) * cs.cos(beta[:-1])
                                                        + (F_yfl + F_yfr) * cs.cos(delta - beta[:-1])
                                                        + F_d * cs.sin(beta[:-1])))
            g.append(beta_dot_cst)
            ubg.append([0.0] * n)
            lbg.append([0.0] * n)

            # --- dot_omega = ...
            omega_dot_cst = kappa[1:] * v[1:] \
                - kappa[1:] * v[:-1] - ds / v[: - 1] \
                            * (1 / I_zz * (+ (F_xrr - F_xrl) * tw_r / 2 - (F_yrl + F_yrr)
                                           * l_r + ((F_xfr - F_xfl) * cs.cos(delta)
                                                    + (F_yfl - F_yfr) * cs.sin(delta))
                                           * tw_f / 2 + ((F_yfl + F_yfr) * cs.cos(delta)
                                                         + (F_xfl + F_xfr) * cs.sin(delta)) * l_f))
            g.append(omega_dot_cst)
            ubg.append([0.0] * n)
            lbg.append([0.0] * n)

            # --- gamma = ...
            gamma_cst = gamma_y - h_cg / (0.5 * (tw_f + tw_r)) * ma_y
            g.append(gamma_cst)
            ubg.append([0.0] * n)
            lbg.append([0.0] * n)

            ############################################################################################################
            # Inequality Constraints
            ############################################################################################################

            # --- 0 <= v * F_dr <= P_max
            if self.sol_dict[self.key]['VarPower'] is False:
                pwr_cst = F_dr * v[:-1]
                g.append(pwr_cst)
                ubg.append([P_max] * n)
                lbg.append([-np.inf] * n)
            else:
                pwr_cst = F_dr * v[:-1] - P_max_param[0:]
                g.append(pwr_cst)
                ubg.append([0] * n)
                lbg.append([-np.inf] * n)

            # --- -0.02 <= F_br * F_dr <= 0
            F_dr_F_br_cst = F_br * F_dr
            g.append(F_dr_F_br_cst)
            ubg.append([0.0] * n)
            lbg.append([-0.02] * n)

            # --- Friction circle
            mu_x = ax_max / a_max
            mu_y = ay_max / a_lat_max

            # Front tire left
            fric_front_left_tire_cst = (F_xfl / (mu_x * F_zfl)) ** 2 + (F_yfl / (mu_y * F_zfl)) ** 2
            g.append(fric_front_left_tire_cst)
            ubg.append([1.0] * n)
            lbg.append([0.0] * n)

            # Front tire right
            fric_front_right_tire_cst = (F_xfr / (mu_x * F_zfr)) ** 2 + (F_yfr / (mu_y * F_zfr)) ** 2
            g.append(fric_front_right_tire_cst)
            ubg.append([1.0] * n)
            lbg.append([0.0] * n)

            # Rear tire left
            fric_rear_left_tire_cst = (F_xrl / (mu_x * F_zrl)) ** 2 + (F_yrl / (mu_y * F_zrl)) ** 2
            g.append(fric_rear_left_tire_cst)
            ubg.append([1.0] * n)
            lbg.append([0.0] * n)

            # Rear tire right
            fric_rear_right_tire_cst = (F_xrr / (mu_x * F_zrr)) ** 2 + (F_yrr / (mu_y * F_zrr)) ** 2
            g.append(fric_rear_right_tire_cst)
            ubg.append([1.0] * n)
            lbg.append([0.0] * n)

            # Change Rate of Driving Force
            dot_f_dr_cst = (F_dr[1:] - F_dr[:-1]) / (ds[:-1] / v[:-2])
            g.append(dot_f_dr_cst)
            ubg.append([F_max] * (m - 2))
            lbg.append([-np.inf] * (m - 2))

            # Change Rate of Braking Force
            dot_f_br_cst = (F_br[1:] - F_br[:-1]) / (ds[:-1] / v[:-2])
            g.append(dot_f_br_cst)
            ubg.append([np.inf] * (m - 2))
            lbg.append([F_min] * (m - 2))

            # Change Rate of Steering Angle
            dot_delta_cst = (delta[1:] - delta[:-1]) / (ds[:-1] / v[:-2])
            g.append(dot_delta_cst)
            ubg.append([delta_max] * (m - 2))
            lbg.append([- delta_max] * (m - 2))

            ############################################################################################################
            # Solver stuff
            ############################################################################################################
            # Initialization of optimization variables, OVERRIDDEN during usage
            x0.append([10] * m)  # initial guess for velocity
            x0.append([0] * m)  # initial guess for slip angle
            x0.append([0] * n)  # initial guess for driving force
            x0.append([0] * n)  # iitial guess for braking force
            x0.append([0] * n)  # initial guess for steering rate
            x0.append([0] * n)  # initial guess for lateral tire distribution

            x0 = np.concatenate(x0)

            # Formatting lower and upper bounds of optimization variables if available
            # Formatting constraints
            g = cs.vertcat(*g)
            lbg = cs.vertcat(*lbg)
            ubg = cs.vertcat(*ubg)

            # --- Formatting parameter vector for optimization
            # Constant friction
            if self.sol_dict[self.key]['VarPower'] is False:
                param_vec = cs.vertcat(kappa,
                                       ds,
                                       v_ini_param,
                                       v_end_param,
                                       ax_max,
                                       ay_max)
            else:
                param_vec = cs.vertcat(kappa,
                                       ds,
                                       v_ini_param,
                                       v_end_param,
                                       P_max_param,
                                       ax_max,
                                       ay_max)

            # Formatting optimization vector for optimization
            x = cs.vertcat(v, beta, F_dr, F_br, delta, gamma_y)  # create optimization variables vector

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

            self.trj = []
            self.x = x
            self.x0 = x0
            self.lbx = lbx
            self.ubx = ubx
            self.g = g
            self.lbg = lbg
            self.ubg = ubg
            self.J = 0

            pass

    def calc_v_ipopt(self,
                     v_ini: np.ndarray,
                     kappa: np.ndarray,
                     delta_s: np.ndarray,
                     v_max: np.ndarray,
                     F_ini: float,
                     v_end: float,
                     P_max: np.array,
                     x0_v: np.ndarray,
                     x0_s_t: np.ndarray,
                     ax_max: np.ndarray,
                     ay_max: np.ndarray) -> tuple:
        """Function to update the paramter vector and initial guess for the solution.

        :param v_ini: initial hard constrained velocity [m/s]
        :param kappa: curvature profile of given path [rad/m]
        :param delta_s: discretization step length of given path [m]
        :param v_max: max. allowed velocity (in objective function) [m/s]
        :param F_ini: hard constrained initial force [kN]
        :param v_end: hard constrained max. allowed value of end velocity in optimization horizon [m/s]
        :param P_max: max. allowed power [kW]
        :param x0_v: initial guess velocity [m/s]
        :param x0_s_t: initial guess slack variables tire [-]
        :param ax_max: max. allowed longitudinal acceleration [m/s^2]
        :param ay_max: max. allowed lateral accelereation [m/s]

        :return: sol: solution of the QP \n
            param_vec: parameter vector \n
            dt_ipopt: runtime of the solver IPOPT [ms] \n
            sol_status: status of the solution (solved, infeasible, etc.)

        :Authors:
            Thomas Herrmann <thomas.herrmann@tum.de> \n
            Tobias Klotz <tobias.klotz@tum.de>

        :Created on:
            01.01.2020
        """

        # --- Parameter vector for NLP
        param_vec = list()

        # Point Mass Model
        if self.sol_dict[self.key]['Model'] == "PM":
            param_vec.append([v_ini])
            param_vec.append(kappa)
            param_vec.append(delta_s)
            param_vec.append(v_max)
            param_vec.append([F_ini])
            param_vec.append([v_end])
            if self.sol_dict[self.key]['VarPower'] is False:
                pass
            else:
                param_vec.append(P_max)
            if self.sol_dict[self.key]['VarFriction']:
                param_vec.append(ax_max)
                param_vec.append(ay_max)

            # Conversion of format
            param_vec = np.concatenate(param_vec)

            # --- Initial guess for NLP
            x0_nlp = list()
            x0_nlp.append(x0_v)
            if self.sol_dict[self.key]['Slack']:
                x0_nlp.append(x0_s_t)

            # Conversion of format
            x0_nlp = np.concatenate(x0_nlp)

        # Kinematic bicycle model
        elif self.sol_dict[self.key]['Model'] == "KM":
            param_vec.append([v_ini])
            param_vec.append(kappa)
            param_vec.append(delta_s)
            param_vec.append(v_max)
            param_vec.append([F_ini])
            param_vec.append([v_end])
            if self.sol_dict[self.key]['VarPower'] is False:
                pass
            else:
                param_vec.append(P_max)
            if self.sol_dict[self.key]['VarFriction']:
                param_vec.append(ax_max)
                param_vec.append(ay_max)

            # Conversion of format
            param_vec = np.concatenate(param_vec)

            # --- Initial guess for NLP
            x0_nlp = list()
            if x0_v[0] == 0:
                x0_v[0] = 0.1
            x0_nlp.append(x0_v)
            if self.sol_dict[self.key]['Slack']:
                x0_nlp.append(x0_s_t)

            # Conversion of format
            x0_nlp = np.concatenate(x0_nlp)

        # Dynamic bicycle model
        elif self.sol_dict[self.key]['Model'] == "DM":
            param_vec.append(kappa[0:self.m])
            param_vec.append(delta_s)
            if v_ini == 0:
                v_ini = 1
            param_vec.append([v_ini])
            param_vec.append([v_end])
            if self.sol_dict[self.key]['VarPower'] is False:
                pass
            else:
                param_vec.append(P_max)
            if ax_max is not None:
                param_vec.append(ax_max)
                param_vec.append(ay_max)

            # Conversion of format
            param_vec = np.concatenate(param_vec)

            x0_nlp = list()

            # --- Initial guess for NLP
            v0 = x0_v
            if v0[0] == 0:
                v0[0] = 1
            beta0 = np.zeros(self.m)
            omega0 = np.zeros(self.m)
            F_dr0 = np.zeros(self.m - 1)
            F_br0 = np.zeros(self.m - 1)
            delta0 = np.zeros(self.m - 1)
            x0_nlp.append(v0)
            x0_nlp.append(beta0)
            x0_nlp.append(omega0)
            x0_nlp.append(F_dr0)
            x0_nlp.append(F_br0)
            x0_nlp.append(delta0)

            # Conversion of format
            x0_nlp = np.concatenate(x0_nlp)

        # Fourwheel model
        elif self.sol_dict[self.key]['Model'] == "FW":
            param_vec.append(kappa[0:self.m])
            param_vec.append(delta_s)
            if v_ini == 0:
                v_ini = 1
            param_vec.append([v_ini])
            param_vec.append([v_end])
            if self.sol_dict[self.key]['VarPower'] is False:
                pass
            else:
                param_vec.append(P_max)
            if ax_max is not None:
                param_vec.append(ax_max)
                param_vec.append(ay_max)

            # Conversion of format
            param_vec = np.concatenate(param_vec)

            x0_nlp = list()

            # --- Initial guess for NLP
            v0 = x0_v
            if v0[0] == 0:
                v0[0] = 1
            beta0 = np.zeros(self.m)
            F_dr0 = np.zeros(self.m - 1)
            F_br0 = np.zeros(self.m - 1)
            delta0 = np.zeros(self.m - 1)
            gamma0 = np.zeros(self.m - 1)
            x0_nlp.append(v0)
            x0_nlp.append(beta0)
            x0_nlp.append(F_dr0)
            x0_nlp.append(F_br0)
            x0_nlp.append(delta0)
            x0_nlp.append(gamma0)

            # Conversion of format
            x0_nlp = np.concatenate(x0_nlp)

        ################################################################################################################
        # Solve NLP
        ################################################################################################################
        sol, dt_ipopt, sol_status = self.solve(x0=x0_nlp,
                                               param_vec=param_vec)

        return sol, param_vec, dt_ipopt, sol_status

    def solve(self,
              x0: np.ndarray,
              param_vec: np.ndarray) -> tuple:
        """Function to solve the constructed QP.

        :param x0: initial guess of the solution
        :param param_vec: paramter vector

        :return: sol: solution of the QP \n
            dt_ipopt: runtime of the Solver IPOPT [ms] \n
            sol_status: status of the solution (solved, infeasible, etc.)

        :Authors:
            Thomas Herrmann <thomas.herrmann@tum.de> \n
            Tobias Klotz <tobias.klotz@tum.de>

        :Created on:
            01.01.2020
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

        dt_ipopt = (t1 - t0) * 1000
        sol_status = solver.stats()['return_status']
        return sol, dt_ipopt, sol_status

    def transform_sol(self,
                      sol: dict,
                      param_vec_: np.ndarray,
                      vis_options: dict) -> tuple:
        """Function to recalculat the optimization variables of the QP.

        :param sol: solution of the QP
        :param param_vec: paramter vector
        :param vis_options: user specified visualization options of the debugging tool

        :return: v: optimized velocity [m/s] \n
            eps_tre: optimized tire slacks [-] \n
            F_p: optimize powertrain force [kN] \n
            P_p: optimized power force [kW] \n
            a_x: acceleration in x-direction of CoG [m/s²] \n
            a_y: acceleration in y-direction of CoG [m/s²]

        :Authors:
            Thomas Herrmann <thomas.herrmann@tum.de> \n
            Tobias Klotz <tobias.klotz@tum.de>

        :Created on:
            01.01.2020
        """
        opt_config = configparser.ConfigParser()
        if not opt_config.read(self.params_path + 'sqp_config.ini'):
            raise ValueError('Specified cost config file does not exist or is empty!')
        m = self.m
        trj = self.trj

        # vehicle mass [t]
        m_t = opt_config.getfloat('VEHICLE', 'mass_t')
        c_res = opt_config.getfloat('VEHICLE', 'c_res')

        # Initialization of output vector
        eps_tre = []
        F_xf = []
        F_xr = []
        F_yf = []
        F_yr = []

        F_xfl = []
        F_xfr = []
        F_xrl = []
        F_xrr = []
        F_yfl = []
        F_yfr = []
        F_yrl = []
        F_yrr = []

        ################################################################################################################
        # Retrieve solution
        ################################################################################################################
        # --- Extract variables
        # Point mass model
        if self.sol_dict[self.key]['Model'] == "PM":
            # Extract velocity
            v = np.array(sol['x'][0:m])
            # Extract slack variable
            if self.sol_dict[self.key]['Slack']:
                eps_tre = np.array(sol['x'][m:m + int(np.ceil(m / self.slack_every_v))])

            # Calculate engine force
            F_p, tire_cst, a, pwr_cst = trj(sol['x'], param_vec_)
            # Calculate engine power
            P_p = F_p * v[0:-1]

            # Calculate Acceleration ax
            kappa = param_vec_[1:self.m + 1]
            delta_s = param_vec_[self.m + 1:2 * self.m]

            ax = []
            ay = []
            for k in range(self.m - 1):
                ax.append(F_p[k] / m_t)
                ay.append(kappa[k] * v[k] ** 2)
            ax = np.array(ax)

            return v, eps_tre, np.array(F_p), P_p, ax, ay, F_xf, F_yf, F_xr, F_yr, F_xfl, F_xfr, F_yfl, F_yfr, F_xrl, \
                F_xrr, F_yrl, F_yrr

        # Kinematic bicycle model
        elif self.sol_dict[self.key]['Model'] == "KM":
            # Extract velocity
            v = np.array(sol['x'][0:m])
            # Extract slack variable
            if self.sol_dict[self.key]['Slack']:
                eps_tre = np.array(sol['x'][m:m + int(np.ceil(m / self.slack_every_v))])

            # Calculate engine force and acceleartion
            kappa = param_vec_[1:self.m + 1]
            delta_s = param_vec_[self.m + 1:2 * self.m]
            acc = []
            ax = []
            ay = []
            F_p = []
            for k in range(self.m - 1):
                acc.append((v[k + 1] ** 2 - v[k] ** 2) / (2 * delta_s[k]))
                ay.append(kappa[k] * v[k] ** 2)
                F_p.append(m_t * acc[k] - c_res * 0.001 * v[k] ** 2)
                ax.append(F_p[k] / m_t)
            # Calculate engine power
            P_p = F_p * v[0:-1]

            return v, eps_tre, np.array(F_p), P_p, ax, ay, F_xf, F_yf, F_xr, F_yr, F_xfl, F_xfr, F_yfl, F_yfr, F_xrl, \
                F_xrr, F_yrl, F_yrr

        # Dynamic bicycle model
        elif self.sol_dict[self.key]['Model'] == "DM":
            # elif self.model == 'dynamic_bicycle':
            # --- Extract variables
            v = np.array(sol['x'][0:m])
            beta = np.array(sol['x'][m:2 * m])
            omega = np.array(sol['x'][2 * m:3 * m])
            F_dr = np.array(sol['x'][3 * m:4 * m - 1])
            F_br = np.array(sol['x'][4 * m - 1:5 * m - 2])
            delta = np.array(sol['x'][5 * m - 2:6 * m - 3])

            v = np.concatenate(v)
            beta = np.concatenate(beta)
            omega = np.concatenate(omega)
            F_dr = np.concatenate(F_dr)
            F_br = np.concatenate(F_br)
            delta = np.concatenate(delta)

            # Calculate engine force from driving and braking force [kN]
            F_p = F_dr + F_br
            # Calculate engine power [kW]
            P_p = F_p * v[0: - 1]

            kappa = param_vec_[0:self.m]
            # Calculate Forces fto calculate accleration at CoG
            ############################################################################################################
            # Forces
            ############################################################################################################
            # gravittational constatns [m/s²]
            grav = opt_config.getfloat('CAR_PARAMETER', 'g')
            # rolling friction coefficient[-]
            f_r = opt_config.getfloat('CAR_PARAMETER', 'f_r')
            # length from CoG to Front Axle [m]
            l_f = opt_config.getfloat('CAR_PARAMETER', 'l_f')
            # length from CoG to Rear Axle [m]
            l_r = opt_config.getfloat('CAR_PARAMETER', 'l_r')
            # Front Surface [m²]
            A = opt_config.getfloat('CAR_PARAMETER', 'A')
            # Air Density [kg/m³]
            rho = opt_config.getfloat('CAR_PARAMETER', 'rho')
            # Distribution of Engine Force (Front/Rear) [-]
            k_dr = opt_config.getfloat('CAR_PARAMETER', 'k_dr')
            # Distribution of Braking Force (Front/Rear) [-]
            k_br = opt_config.getfloat('CAR_PARAMETER', 'k_br')
            # Drift Coefficient at Front Tire [-]
            c_lf = opt_config.getfloat('CAR_PARAMETER', 'c_lf')
            # Drift Coefficient at Rear Tire [-]
            c_lr = opt_config.getfloat('CAR_PARAMETER', 'c_lr')
            # Height of CoG [m]
            h_cg = opt_config.getfloat('CAR_PARAMETER', 'h_cg')

            m_t = opt_config.getfloat('VEHICLE', 'mass_t')

            # Tire Model (Magic Formula)
            F_z0 = opt_config.getfloat('CAR_PARAMETER', 'F_z0')  # [kN]
            B_f = opt_config.getfloat('CAR_PARAMETER', 'B_f')
            C_f = opt_config.getfloat('CAR_PARAMETER', 'C_f')
            D_f = opt_config.getfloat('CAR_PARAMETER', 'D_f')
            E_f = opt_config.getfloat('CAR_PARAMETER', 'E_f')
            eps_f = opt_config.getfloat('CAR_PARAMETER', 'eps_f')
            B_r = opt_config.getfloat('CAR_PARAMETER', 'B_r')
            C_r = opt_config.getfloat('CAR_PARAMETER', 'C_r')
            D_r = opt_config.getfloat('CAR_PARAMETER', 'D_r')
            E_r = opt_config.getfloat('CAR_PARAMETER', 'E_r')
            eps_r = opt_config.getfloat('CAR_PARAMETER', 'eps_r')

            # front and rear roll friction
            F_roll_f = f_r * m_t * grav * l_r / (l_r + l_f)
            F_roll_r = f_r * m_t * grav * l_f / (l_r + l_f)

            alpha_f = delta - cs.atan2((l_f * kappa[: - 1] * v[: - 1] / (2 * np.pi) + v[: - 1] * cs.sin(beta[: - 1])),
                                       v[: - 1] * cs.cos(beta[: - 1]))
            alpha_r = cs.atan2(l_r * kappa[: - 1] * v[: - 1] / (2 * np.pi) - v[: - 1] * cs.sin(beta[: - 1]),
                               v[: - 1] * cs.cos(beta[: - 1]))

            F_d = 0.001 * c_res * v[: - 1] ** 2

            F_xf = k_dr * F_dr + k_br * F_br - F_roll_f
            F_xr = (1 - k_dr) * F_dr + (1 - k_br) * F_br - F_roll_r

            ma_x = F_xf + F_xr - F_d

            F_zf = m_t * grav * l_r / (l_r + l_f) - h_cg \
                / (l_r + l_f) * ma_x + 0.5 * c_lf * rho * A * v[: - 1] ** 2
            F_zr = m_t * grav * l_f / (l_r + l_f) + h_cg \
                / (l_r + l_f) * ma_x + 0.5 * c_lr * rho * A * v[: - 1] ** 2

            F_yf = D_f * (1 + eps_f * F_zf / F_z0) * F_zf / F_z0 * cs.sin(
                C_f * cs.atan2(B_f * alpha_f - E_f * (B_f * alpha_f - cs.atan2(B_f * alpha_f, 1)), 1))
            F_yr = D_r * (1 + eps_r * F_zr / F_z0) * F_zr / F_z0 * cs.sin(
                C_r * cs.atan2(B_r * alpha_r - E_r * (B_r * alpha_r - cs.atan2(B_r * alpha_r, 1)), 1))

            ma_y = F_yr + F_xf * cs.sin(delta) + F_yf * cs.cos(delta)

            ax = []
            ay = []
            for k in range(self.m - 1):
                ax.append(ma_x[k] / m_t)
                ay.append(ma_y[k] / m_t)

            ax = np.array(ax)
            ay = np.array(ay)

            '''# --- Friction circle
            mu_x = ax_max / a_max
            mu_y = ay_max / a_lat_max'''

            # Front tire Constraint
            F_xzf = (F_xf / F_zf)
            F_yzf = (F_yf / F_zf)
            # Rear tire Constraint
            F_xzr = (F_xr / F_zr)
            F_yzr = (F_yr / F_zr)
            '''
            fric_front_tire_cst = (F_xf / (mu_x * F_zf)) ** 2 + (F_yf / (mu_y * F_zf)) ** 2
            g.append(fric_front_tire_cst)
            ubg.append([1.0] * n)
            lbg.append([0.0] * n)

            # Rear tire
            fric_rear_tire_cst = (F_xr / (mu_x * F_zr)) ** 2 + (F_yr / (mu_y * F_zr)) ** 2
            g.append(fric_rear_tire_cst)
            ubg.append([1.0] * n)
            lbg.append([0.0] * n)'''

            return v, eps_tre, np.array(F_p), P_p, ax, ay, np.array(F_xzf), np.array(F_yzf), np.array(F_xzr), \
                np.array(F_yzr), F_xfl, F_xfr, F_yfl, F_yfr, F_xrl, F_xrr, F_yrl, F_yrr

        elif self.sol_dict[self.key]['Model'] == "FW":
            # --- Extract variables
            v = np.array(sol['x'][0:m])
            beta = np.array(sol['x'][m:2 * m])
            F_dr = np.array(sol['x'][2 * m:3 * m - 1])
            F_br = np.array(sol['x'][3 * m - 1:4 * m - 2])
            delta = np.array(sol['x'][4 * m - 2:5 * m - 3])
            gamma_y = np.array(sol['x'][5 * m - 3:6 * m - 4])
            v = np.concatenate(v)
            beta = np.concatenate(beta)
            F_dr = np.concatenate(F_dr)
            F_br = np.concatenate(F_br)
            delta = np.concatenate(delta)
            gamma_y = np.concatenate(gamma_y)

            kappa = param_vec_[0:self.m]

            # Calculate ower froce [kN]
            F_p = F_dr + F_br
            # Calculate engine power [kW]
            P_p = F_p * v[0:-1]

            ############################################################################################################
            # Forces
            ############################################################################################################
            # gravittational constatns [m/s²]
            grav = opt_config.getfloat('CAR_PARAMETER', 'g')
            # rolling friction coefficient[-]
            f_r = opt_config.getfloat('CAR_PARAMETER', 'f_r')
            # length from CoG to Front Axle [m]
            l_f = opt_config.getfloat('CAR_PARAMETER', 'l_f')
            # length from CoG to Rear Axle [m]
            l_r = opt_config.getfloat('CAR_PARAMETER', 'l_r')
            # Front Surface [m²]
            A = opt_config.getfloat('CAR_PARAMETER', 'A')
            # Air Density [kg/m³]
            rho = opt_config.getfloat('CAR_PARAMETER', 'rho')
            # Distribution of Engine Force (Front/Rear) [-]
            k_dr = opt_config.getfloat('CAR_PARAMETER', 'k_dr')
            # Distribution of Braking Force (Front/Rear) [-]
            k_br = opt_config.getfloat('CAR_PARAMETER', 'k_br')

            k_roll = opt_config.getfloat('CAR_PARAMETER', 'k_roll')
            # Drift Coefficient at Front Tire [-]
            c_lf = opt_config.getfloat('CAR_PARAMETER', 'c_lf')
            # Drift Coefficient at Rear Tire [-]
            c_lr = opt_config.getfloat('CAR_PARAMETER', 'c_lr')
            # Height of CoG [m]
            h_cg = opt_config.getfloat('CAR_PARAMETER', 'h_cg')

            tw_f = opt_config.getfloat('CAR_PARAMETER', 'tw_f')
            tw_r = opt_config.getfloat('CAR_PARAMETER', 'tw_r')

            m_t = opt_config.getfloat('VEHICLE', 'mass_t')

            # Tire Model (Magic Formula)
            F_z0 = opt_config.getfloat('CAR_PARAMETER', 'F_z0')  # [kN]
            B_f = opt_config.getfloat('CAR_PARAMETER', 'B_f')
            C_f = opt_config.getfloat('CAR_PARAMETER', 'C_f')
            D_f = opt_config.getfloat('CAR_PARAMETER', 'D_f')
            E_f = opt_config.getfloat('CAR_PARAMETER', 'E_f')
            eps_f = opt_config.getfloat('CAR_PARAMETER', 'eps_f')
            B_r = opt_config.getfloat('CAR_PARAMETER', 'B_r')
            C_r = opt_config.getfloat('CAR_PARAMETER', 'C_r')
            D_r = opt_config.getfloat('CAR_PARAMETER', 'D_r')
            E_r = opt_config.getfloat('CAR_PARAMETER', 'E_r')
            eps_r = opt_config.getfloat('CAR_PARAMETER', 'eps_r')

            # roll friction (front & rear/ left & right) [kN]
            F_roll_fl = 0.5 * f_r * m_t * grav * l_r / (l_r + l_f)
            F_roll_fr = 0.5 * f_r * m_t * grav * l_r / (l_r + l_f)
            F_roll_rl = 0.5 * f_r * m_t * grav * l_f / (l_r + l_f)
            F_roll_rr = 0.5 * f_r * m_t * grav * l_f / (l_r + l_f)

            # tire slip angle (front & rear/ left & right) [rad]
            alpha_fl = delta - cs.atan2((l_f * kappa[: - 1] * v[: - 1] / (2 * np.pi) + v[: - 1] * cs.sin(beta[: - 1])),
                                        v[: - 1] * cs.cos(beta[: - 1]) - 0.5 * tw_f * kappa[: - 1] * v[: - 1]
                                        / (2 * np.pi))
            alpha_fr = delta - cs.atan2((l_f * kappa[: - 1] * v[: - 1] / (2 * np.pi) + v[: - 1] * cs.sin(beta[: - 1])),
                                        v[: - 1] * cs.cos(beta[: - 1]) + 0.5 * tw_f * kappa[: - 1] * v[: - 1]
                                        / (2 * np.pi))

            alpha_rl = cs.atan2(l_r * kappa[: - 1] * v[: - 1] / (2 * np.pi) - v[: - 1] * cs.sin(beta[: - 1]),
                                v[: - 1] * cs.cos(beta[: - 1]) - 0.5 * tw_r * kappa[: - 1] * v[: - 1] / (2 * np.pi))
            alpha_rr = cs.atan2(l_r * kappa[: - 1] * v[: - 1] / (2 * np.pi) - v[: - 1] * cs.sin(beta[: - 1]),
                                v[: - 1] * cs.cos(beta[: - 1]) + 0.5 * tw_r * kappa[: - 1] * v[: - 1] / (2 * np.pi))

            # air resistance [kN]
            F_d = 0.001 * c_res * v[: - 1] ** 2

            # tire force in x-direction (front & rear/ left & right) [kN]
            F_xfl = 0.5 * k_dr * F_dr + 0.5 * k_br * F_br - F_roll_fl
            F_xfr = 0.5 * k_dr * F_dr + 0.5 * k_br * F_br - F_roll_fr

            F_xrl = 0.5 * (1 - k_dr) * F_dr + 0.5 * (1 - k_br) * F_br - F_roll_rl
            F_xrr = 0.5 * (1 - k_dr) * F_dr + 0.5 * (1 - k_br) * F_br - F_roll_rr

            # total force in x-direction at CoG [kN]
            ma_x = F_xrl + F_xrr + (F_xfl + F_xfr) - F_d

            # tire normal force (front & rear / left & right) [kN]
            F_zfl = + 0.5 * m_t * grav * l_r / (l_r + l_f) - 0.5 * h_cg / (l_r + l_f) * ma_x - k_roll * gamma_y \
                    + 0.25 * c_lf * rho * A * v[: - 1] ** 2

            F_zfr = + 0.5 * m_t * grav * l_r / (l_r + l_f) - 0.5 * h_cg / (l_r + l_f) * ma_x \
                    + k_roll * gamma_y \
                    + 0.25 * c_lf * rho * A * v[: - 1] ** 2

            F_zrl = + 0.5 * m_t * grav * l_f / (l_r + l_f) \
                    + 0.5 * h_cg / (l_r + l_f) * ma_x - (1 - k_roll) * gamma_y \
                    + 0.25 * c_lr * rho * A * v[: - 1] ** 2

            F_zrr = + 0.5 * m_t * grav * l_f / (l_r + l_f) \
                    + 0.5 * h_cg / (l_r + l_f) * ma_x \
                    + (1 - k_roll) * gamma_y \
                    + 0.25 * c_lr * rho * A * v[: - 1] ** 2

            # tire force in y-direction (front & rear/ left & right) [kN]
            F_yfl = D_f * (1 + eps_f * F_zfl / F_z0) * F_zfl / F_z0 \
                * cs.sin(C_f * cs.atan2(B_f * alpha_fl - E_f * (B_f * alpha_fl - cs.atan2(B_f * alpha_fl, 1)), 1))
            F_yfr = D_f * (1 + eps_f * F_zfr / F_z0) * F_zfr / F_z0 \
                * cs.sin(C_f * cs.atan2(B_f * alpha_fr - E_f * (B_f * alpha_fr - cs.atan2(B_f * alpha_fr, 1)), 1))

            F_yrl = D_r * (1 + eps_r * F_zrl / F_z0) * F_zrl / F_z0 \
                * cs.sin(C_r * cs.atan2(B_r * alpha_rl - E_r * (B_r * alpha_rl - cs.atan2(B_r * alpha_rl, 1)), 1))
            F_yrr = D_r * (1 + eps_r * F_zrr / F_z0) * F_zrr / F_z0 \
                * cs.sin(C_r * cs.atan2(B_r * alpha_rr - E_r * (B_r * alpha_rr - cs.atan2(B_r * alpha_rr, 1)), 1))

            # total force in y-direction at CoG
            ma_y = F_yrl + F_yrr + (F_xfl + F_xfr) * cs.sin(delta) + (F_yfl + F_yfr) * cs.cos(delta)

            ax = ma_x / m_t
            ay = ma_y / m_t
            ax = []
            ay = []
            for k in range(self.m - 1):
                ax.append(ma_x[k] / m_t)
                ay.append(ma_y[k] / m_t)

            ax = np.array(ax)
            ay = np.array(ay)

            # Front left tire Constraint
            F_xzfl = (F_xfl / F_zfl)
            F_yzfl = (F_yfl / F_zfl)
            # Front right tire Constraint
            F_xzfr = (F_xfr / F_zfr)
            F_yzfr = (F_yfr / F_zfr)

            # Rear left tire Constraint
            F_xzrl = (F_xrl / F_zrl)
            F_yzrl = (F_yrl / F_zrl)
            # Rear right tire Constraint
            F_xzrr = (F_xrr / F_zrr)
            F_yzrr = (F_yrr / F_zrr)

            return v, eps_tre, np.array(F_p), P_p, ax, ay, F_xf, F_yf, F_xr, F_yr, np.array(F_xzfl), np.array(F_xzfr), \
                np.array(F_yzfl), np.array(F_yzfr), np.array(F_xzrl), np.array(F_xzrr), np.array(F_yzrl), \
                np.array(F_yzrr)


if __name__ == '__main__':
    pass
