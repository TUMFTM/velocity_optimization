try:
    import casadi as cs
except ImportError:
    print('Warning: No module CasADi found. Not necessary on car but for development.')
try:
    import sympy as sym
except ImportError:
    print('Warning: No module sympy found. Not necessary on car but for development.')
try:
    import dill
except ImportError:
    print('Warning: No module dill found. Not necessary on car but for development.')
import os
import sys
import numpy as np
import time
import configparser
from scipy import sparse
import matplotlib.pyplot as plt


class VOpt_qpOASES:

    __slots__ = 'solver'

    def __init__(self,
                 Hm: np.ndarray,
                 Am: np.ndarray):
        """Class to optimize a velocity profile for a given path using the solver qpOASES.

        .. math::
            \min_x \qquad 1/2~x^T H_m x + q^T_v x \n
            \mathrm{s.t} \qquad lba \leq A_m x \leq uba

        :param Hm: Hessian problem matrix
        :param Am: Linearized constraints matrix (Jacobian)

        :Authors:
            Thomas Herrmann <thomas.herrmann@tum.de>

        :Created on:
            01.01.2020
        """

        self.solver = None

        # --- Initialization of qpOASES solver object
        self.sol_init(Hm=Hm,
                      Am=Am)

    def sol_init(self,
                 Hm: np.ndarray,
                 Am: np.ndarray):
        """Function to initialize the qpOASES solver.

        :param Hm: Hessian problem matrix
        :param Am: Linearized constraints matrix (Jacobian)

        :Authors:
            Thomas Herrmann <thomas.herrmann@tum.de>

        :Created on:
            01.01.2020
        """

        opts_qpOASES = {"terminationTolerance": 1e-2,
                        "printLevel": "low",
                        "hessian_type": "posdef",
                        "error_on_fail": False,
                        "sparse": True}

        # --- Create solver size
        Hm = cs.DM(Hm)
        Am = cs.DM(Am)

        # --- Initialize QP-structure
        QP = dict()
        QP['h'] = Hm.sparsity()
        QP['a'] = Am.sparsity()

        self.solver = cs.conic('solver', 'qpoases', QP, opts_qpOASES)

    def solve(self,
              x0: np.ndarray,
              Hm: np.ndarray,
              gv: np.ndarray,
              Am: np.ndarray,
              lba: np.ndarray,
              uba: np.ndarray) -> list:
        """Function to solve qpOASES optimization problem.

        :param x0: initial guess of optimization variables,
        :param Hm: Hessian problem matrix
        :param gv: Jacobian of problem's objective function,
        :param Am: Linearized constraints matrix
        :param lba: lower boundary vector constraints
        :param uba: upper boundary vector constraints

        :return: x_opt: optimized qpOASES solution vector

        :Authors:
            Thomas Herrmann <thomas.herrmann@tum.de>

        :Created on:
            01.01.2020
        """

        # Hessian is constant, no need to overwrite
        # Hm = DM(Hm)
        Am = cs.DM(Am)
        gv = cs.DM(gv)

        # --- Solve QP
        t0 = time.perf_counter()

        # --- Hessian is constant, no need for update
        # r = self.solver(x0=x0, h=Hm, g=gv, a=Am, lba=lba, uba=uba)
        r = self.solver(x0=x0, g=gv, a=Am, lba=lba, uba=uba)

        # --- Re-initialize qpOASES if solver fails
        if not self.solver.stats()['success']:
            self.sol_init(Hm=Hm, Am=Am)

        t1 = time.perf_counter()

        # --- Retrieve optimization variables
        x_opt = r['x']

        print("qpOASES time in ms: ", (t1 - t0) * 1000)

        return x_opt


class Car:

    def __init__(self,
                 params_path: str):
        """Class to initialize the car parameter.

        :param params_path: absolute path to folder containing config file .ini

        :Authors:
            Tobias Klotz <tobias.klotz@tum.de>

        :Created on:
            16.06.2020
        """
        opt_config = configparser.ConfigParser()
        if not opt_config.read(params_path + 'sqp_config.ini'):
            raise ValueError('Specified cost config file does not exist or is empty!')

        # Load Car Paramter
        self.A = opt_config.getfloat('CAR_PARAMETER', 'A')  # [m²] Front Surface
        self.c_lf = opt_config.getfloat('CAR_PARAMETER', 'c_lf')  # [] Drift Coefficient at Front Trie
        self.c_lr = opt_config.getfloat('CAR_PARAMETER', 'c_lr')  # [] Drift Coefficient at Rear Tire
        self.c_w = opt_config.getfloat('CAR_PARAMETER', 'c_w')  # [] Air Resistance Coefficient
        self.f_r = opt_config.getfloat('CAR_PARAMETER', 'f_r')  # [] Rolling Friction Coefficient
        self.h_cg = opt_config.getfloat('CAR_PARAMETER', 'h_cg')  # [m] Height of CoG
        self.I_zz = opt_config.getfloat('CAR_PARAMETER', 'I_zz')  # [t m²] Yaw Inertia Coefficient
        self.k_dr = opt_config.getfloat('CAR_PARAMETER', 'k_dr')  # [] Distribution of Engine Force (Front/Rear)
        self.k_br = opt_config.getfloat('CAR_PARAMETER', 'k_br')  # [] Distribution of Braking Force (Front/Rear)
        self.L = opt_config.getfloat('CAR_PARAMETER', 'L')  # [m] Total Car Length
        self.l_f = opt_config.getfloat('CAR_PARAMETER', 'l_f')  # [m] Length from CoG to Front Axle
        self.l_r = opt_config.getfloat('CAR_PARAMETER', 'l_r')  # [m] Length from CoG to Rear Axle
        self.m = opt_config.getfloat('VEHICLE', 'mass_t')  # [t] vehicle mass
        self.q = opt_config.getfloat('VEHICLE', 'c_res')  # [] Air Resistance Coefficient: 0.5*rho*A*c_w
        self.turn_r = opt_config.getfloat('CAR_PARAMETER', 'turn_r')  # [m] Minimum Turn Radius

        self.delta_max = opt_config.getfloat('CAR_PARAMETER', 'delta_max')  # [rad] Maximum Steer Angle
        self.beta_max = opt_config.getfloat('CAR_PARAMETER', 'beta_max')  # [rad] Maximum Slip Angle
        self.v_delta_max = opt_config.getfloat('CAR_PARAMETER', 'v_delta_max')  # [rad/s] Maximum Steer Angle Rate
        self.omega_max = opt_config.getfloat('CAR_PARAMETER', 'omega_max')  # [rad/s] Maximum Gear Rate
        self.v_min = opt_config.getfloat('CAR_PARAMETER', 'v_min')  # [m/s] Minimum Velocity
        self.v_max = opt_config.getfloat('CAR_PARAMETER', 'v_max')  # [m/s] Maximum Velocity

        self.F_dr_max = opt_config.getfloat('VEHICLE', 'F_max_kN')  # [kN] Maximum Driving Force
        self.F_br_max = opt_config.getfloat('VEHICLE', 'F_min_kN')  # [kN] Maximum Brake Force
        self.P_max = opt_config.getfloat('VEHICLE', 'P_max_kW')  # [kW] Power of Engine

        self.a_lat_max = opt_config.getfloat('VEHICLE', 'ax_max_mps2')  # [m/s²] Maximum Lateral Acceleration
        self.a_max = opt_config.getfloat('VEHICLE', 'ay_max_mps2')  # [m/s²] Maximum Laongitudianl Acceleration
        self.a_long_max = opt_config.getfloat('CAR_PARAMETER', 'a_long_max')
        self.a_long_min = opt_config.getfloat('CAR_PARAMETER', 'a_long_min')
        self.v_end = opt_config.getfloat('CAR_PARAMETER', 'v_end')  # Maximum Velocity at End of Interval

        # Tire Model (Magic Formula)
        self.F_z0 = opt_config.getfloat('CAR_PARAMETER', 'F_z0')  # [kN]
        self.B_f = opt_config.getfloat('CAR_PARAMETER', 'B_f')
        self.C_f = opt_config.getfloat('CAR_PARAMETER', 'C_f')
        self.D_f = opt_config.getfloat('CAR_PARAMETER', 'D_f')
        self.E_f = opt_config.getfloat('CAR_PARAMETER', 'E_f')
        self.eps_f = opt_config.getfloat('CAR_PARAMETER', 'eps_f')
        self.B_r = opt_config.getfloat('CAR_PARAMETER', 'B_r')
        self.C_r = opt_config.getfloat('CAR_PARAMETER', 'C_r')
        self.D_r = opt_config.getfloat('CAR_PARAMETER', 'D_r')
        self.E_r = opt_config.getfloat('CAR_PARAMETER', 'E_r')
        self.eps_r = opt_config.getfloat('CAR_PARAMETER', 'eps_r')

        self.g = opt_config.getfloat('CAR_PARAMETER', 'g')  # [m/s²] Gravitational Constant on Earth
        self.rho = opt_config.getfloat('CAR_PARAMETER', 'rho')  # [kg/m³] Air Density


if __name__ == '__main__':
    pass
