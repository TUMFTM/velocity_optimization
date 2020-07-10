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
import osqp
import configparser
import time
from scipy import sparse
import matplotlib.pyplot as plt


class VOptOSQP:

    def __init__(self,
                 m: int,
                 sid: str,
                 params_path: str,
                 sol_options: dict,
                 key: str):
        """Class to optimize a velocity profile for a given path using the solver OSQP.

        .. math::
            \min_x \qquad 1/2~x^T H_m x + q^T_v x \n
            \mathrm{s.t} \qquad lba \leq A_m x \leq uba

        :param m: number of velocity points
        :param sid: optimized ID 'PerfSQP' or 'EmergSQP'
        :param params_path: absolute path to folder containing config file .ini
        :param sol_options: user specified solver options of the debugging tool
        :param key: key of the used solver

        :Authors:
            Tobias Klotz <tobias.klotz@tum.de>

        :Created on:
            16.06.2020
        """

        self.m = m
        self.sid = sid
        self.F_ini_osqp = []

        self.Car = Car(params_path=params_path)
        self.sol_options = sol_options
        self.key = key

        # select the vehicle dynamic model
        if self.sol_options[self.key]['Model'] == "PM":
            self.sol_init_pm(params_path=params_path)
        if self.sol_options[self.key]['Model'] == "KM":
            self.sol_init_km(params_path=params_path)
        if self.sol_options[self.key]['Model'] == "DM":
            self.sol_init_dm(params_path=params_path)

    # Point mass model
    def sol_init_pm(self,
                    params_path: str):
        """Function to initialize the OSQP solver with the point-mass model \n
        by defining the objective function and constraints.\n
        Saves the matrix and vectors in order to re-use the same QP and avoid recalculations.

        :param params_path: absolute path to folder containing config file .ini

        :Authors:
            Tobias Klotz <tobias.klotz@tum.de>

        :Created on:
            16.06.2020
        """

        # Length of planning horizion
        N = self.m

        # Open Lambdified Functions if they are saved
        mod_local_trajectory_path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        sys.path.append(mod_local_trajectory_path)
        if self.sol_options[self.key]['Friction'] == "Circle":
            file = "Osqp" + "_" + "point_mass" + "_" + str(N) + ".pkl"
        elif self.sol_options[self.key]['Friction'] == "Diamond":
            file = "Osqp" + "_" + "point_mass" + "_" + str(N) + "_" + str(self.sol_options[self.key]['Friction']) \
                   + ".pkl"
        filename = params_path + '/Lambdify_Function/' + file
        try:
            f = open(filename)
            available = True
        except IOError:
            print("File not accessible")
            available = False
        finally:
            if available:
                f.close()

        if available:
            dill.settings['recurse'] = True
            self.P_lam, self.q_lam, self.A_lam, self.lb_lam, self.ub_lam = dill.load(open(filename, "rb"))
            print("Loaded Data")

        else:
            # Decide if Velocity at the beginning and end of the planning horizon is fixed
            vel_start = True
            vel_end = True

            # --- Define Variables ana Parameter:

            ############################################################################################################
            # Optimization Variables
            ############################################################################################################
            # velocity [m/s]
            v = sym.symbols('v0:%d' % (N))
            # Optimization Vector
            var = v

            ############################################################################################################
            # Online Parameters
            ############################################################################################################
            # curvature [1/m]
            kappa = sym.symbols('kappa0:%d' % (N))
            # discretization step length [m]
            ds = sym.symbols('ds:0%d' % (N - 1))
            # initial velocity [m/s]
            v_ini = sym.symbols('v_ini')
            # end velocity [m/s]
            v_end = sym.symbols('v_end')
            # max. velocity [m/s]
            v_max = sym.symbols('v_max0:%d' % (N))
            # initial force [kN]
            F_ini = sym.symbols('F_ini')
            # max. power [kW]
            P_max = sym.symbols('P_max0:%d' % (N - 1))
            # max. acceleration in x-direction of the vehicle [m/s²]
            ax_max = sym.symbols('ax_max0:%d' % (N))
            # max. acceleration in y-direction of the vehicle [m/s²]
            ay_max = sym.symbols('ay_max0:%d' % (N))

            ############################################################################################################
            # Objective function
            ############################################################################################################
            opt_func = 0
            for k in range(1, N):
                opt_func += ds[k - 1] / v[k]

            # Calculate Jacobi Matrix
            q = sym.Matrix([opt_func]).jacobian(var)
            q_lam = sym.lambdify([v, ds], q, 'numpy')

            # Calculate Hessian Matrix
            P = sym.hessian(opt_func, var)
            P_lam = sym.lambdify([v, ds], P, 'numpy')

            ############################################################################################################
            # Constraints
            ############################################################################################################
            h = []
            lb = []
            ub = []
            a = []
            F_dr = []
            # Calculate force and acceleration
            for k in range(N - 1):
                # Calculate acceleration from velocity and step length
                a.append((v[k + 1] ** 2 - v[k] ** 2) / (2 * ds[k]))
                # Driving Force [kN]
                F_dr.append(self.Car.m * a[k] + 0.5 * self.Car.rho * self.Car.c_w * self.Car.A * v[k] ** 2)

            # --- Boundary Conditions
            # Velocity Boundary Constraint
            for k in range(N):
                if k == N - 1 and vel_end:
                    ub.append(v_end - v[k])
                    lb.append(self.Car.v_min - v[k])
                    h.append(v[k])
                elif k == 0 and vel_start:
                    ub.append(v_ini - v[k])
                    lb.append(v_ini - v[k])
                    h.append(v[k])
                else:
                    ub.append(v_max[k] - v[k])
                    lb.append(self.Car.v_min - v[k])
                    h.append(v[k])

            # Driving Force Boundary Constraint
            for k in range(N - 1):
                if k == 0:
                    ub.append(0.0)
                    lb.append(0.0)
                    h.append(F_dr[k] - F_ini)
                else:
                    lb.append(self.Car.F_br_max - F_dr[k])
                    ub.append(self.Car.F_dr_max - F_dr[k])
                    h.append(F_dr[k])

            # Tire Friction
            if self.sol_options[self.key]['Friction'] == "Circle":
                for k in range(N - 1):  # Lateral Acceleration Constraint (CIRCLE)
                    '''lb.append(-np.inf)
                    ub.append(self.Car.a_lat_max ** 2 - a[k] ** 2 - (kappa[k] * v[k] ** 2) ** 2)
                    h.append(a[k] ** 2 + (kappa[k] * v[k] ** 2) ** 2)'''
                    lb.append(-np.inf)
                    ub.append(1 - a[k] ** 2 / ax_max[k] ** 2 - (kappa[k] * v[k] ** 2) ** 2 / ay_max[k] ** 2)
                    h.append(a[k] ** 2 / ax_max[k] ** 2 + (kappa[k] * v[k] ** 2) ** 2 / ay_max[k] ** 2)

                '''acc ** 2 / ax_max ** 2 + (kappa_param[: - 1] * (v[: - 1] ** 2)) ** 2 / ay_max ** 2 - \
                oneone_vec_short'''

            elif self.sol_options[self.key]['Friction'] == "Diamond":
                for k in range(N - 1):  # Lateral Acceleration Constraint (DIAMOND) (1/4)
                    lb.append(- np.inf)
                    ub.append(1 - a[k] / ax_max[k] - (kappa[k] * v[k] ** 2) / ay_max[k])
                    h.append(a[k] / ax_max[k] + (kappa[k] * v[k] ** 2) / ay_max[k])

                for k in range(N - 1):  # Lateral Acceleration Constraint (DIAMOND) (2/4)
                    lb.append(- np.inf)
                    ub.append(1 - a[k] / ax_max[k] + (kappa[k] * v[k] ** 2) / ay_max[k])
                    h.append(a[k] / ax_max[k] - (kappa[k] * v[k] ** 2) / ay_max[k])

                for k in range(N - 1):  # Lateral Acceleration Constraint (DIAMOND) (3/4)
                    lb.append(- np.inf)
                    ub.append(1 + a[k] / ax_max[k] - (kappa[k] * v[k] ** 2) / ay_max[k])
                    h.append(- a[k] / ax_max[k] + (kappa[k] * v[k] ** 2) / ay_max[k])

                for k in range(N - 1):  # Lateral Acceleration Constraint (DIAMOND) (4/4)
                    lb.append(- np.inf)
                    ub.append(1 + a[k] / ax_max[k] + (kappa[k] * v[k] ** 2) / ay_max[k])
                    h.append(- a[k] / ax_max[k] - (kappa[k] * v[k] ** 2) / ay_max[k])

            # Power Constraint
            for k in range(N - 1):
                lb.append(- np.inf)
                ub.append(P_max[k] - F_dr[k] * v[k])
                h.append(F_dr[k] * v[k])

            # Calculate Jacobian Matrix
            print('Calculate Matrix A')
            A = sym.Matrix([h]).jacobian(var)

            print('Lambidfy Matrix')
            A_lam = sym.lambdify([v, F_ini, ds, kappa, v_ini, v_end, v_max, P_max, ax_max, ay_max], A, 'numpy')
            lb_lam = sym.lambdify([v, F_ini, ds, kappa, v_ini, v_end, v_max, P_max, ax_max, ay_max], lb, 'numpy')
            ub_lam = sym.lambdify([v, F_ini, ds, kappa, v_ini, v_end, v_max, P_max, ax_max, ay_max], ub, 'numpy')

            self.P_lam = P_lam
            self.q_lam = q_lam
            self.A_lam = A_lam
            self.lb_lam = lb_lam
            self.ub_lam = ub_lam

            # Save Lambdified_Function
            print('Save Matrix to File')
            mod_local_trajectory_path = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            sys.path.append(mod_local_trajectory_path)
            if self.sol_options[self.key]['Friction'] == "Circle":
                file = "Osqp" + "_" + "point_mass" + "_" + str(N) + ".pkl"
            elif self.sol_options[self.key]['Friction'] == "Diamond":
                file = "Osqp" + "_" + "point_mass" + "_" + str(N) + "_" + str(self.sol_options[self.key]['Friction']) \
                       + ".pkl"

            filepath = params_path + '/Lambdify_Function/'
            filename = filepath + file
            os.makedirs(filepath, exist_ok=True)

            dill.settings['recurse'] = True
            dill.dump([P_lam, q_lam, A_lam, lb_lam, ub_lam], open(filename, "wb"))

    # Kinematic bicycle model
    def sol_init_km(self,
                    params_path: str):
        """Function to initialize the OSQP solver with the kinematic bicycle model \n
        by defining the objective function and constraints.\n
        Saves the matrix and vectors in order to re-use the same QP and avoid recalculations.

        :param params_path: absolute path to folder containing config file .ini

        :Authors:
            Tobias Klotz <tobias.klotz@tum.de>

        :Created on:
            16.06.2020
        """

        # Length of planning horizion
        N = self.m

        # Open Lambdified Functions if they are saved
        if self.sol_options[self.key]['Friction'] == "Circle":
            file = "Osqp" + "_" + "kinematic_bicycle" + "_" + str(N) + ".pkl"
        elif self.sol_options[self.key]['Friction'] == "Diamond":
            file = "Osqp" + "_" + "kinematic_bicycle" + "_" + str(N) + "_" \
                   + str(self.sol_options[self.key]['Friction']) + ".pkl"
        filename = params_path + 'Lambdify_Function/' + file

        try:
            f = open(filename)
            available = True
        except IOError:
            print("File not accessible")
            available = False
        finally:
            if available:
                f.close()

        if available:
            dill.settings['recurse'] = True
            self.P_lam, self.q_lam, self.A_lam, self.lb_lam, self.ub_lam = dill.load(open(filename, "rb"))

        else:
            # Decide if Velocity at the beginning and end of the planning horizon is fixed
            vel_start = True
            vel_end = True

            # --- Define Variables and Parameter:

            ############################################################################################################
            # Optimization Variables
            ############################################################################################################
            # velocity [m/s]
            v = sym.symbols('v0:%d' % (N))
            # Optimization Vector
            var = v  # + delta + F_dr + v_delta

            ############################################################################################################
            # Parameter
            ############################################################################################################
            # curvature [1/m]
            kappa = sym.symbols('kappa0:%d' % (N))
            # discretization step length [m]
            ds = sym.symbols('ds:0%d' % (N - 1))
            # end velocity [m/s]
            v_end = sym.symbols('v_end')
            # max. power [kW]
            P_max = sym.symbols('P_max0:%d' % (N - 1))
            # max. acceleration in x-direction of the vehicle [m/s²]
            ax_max = sym.symbols('ax_max0:%d' % (N))
            # max. acceleration in y-direction of the vehicle [m/s²]
            ay_max = sym.symbols('ay_max0:%d' % (N))

            ############################################################################################################
            # Objective function
            ############################################################################################################
            opt_func = 0
            for k in range(1, N):
                opt_func += ds[k - 1] / v[k]

            # Calculate Jacobi Matrix
            q = sym.Matrix([opt_func]).jacobian(var)
            q_lam = sym.lambdify([v, ds], q, 'numpy')

            # Calculate Hessian Matrix
            P = sym.hessian(opt_func, var)
            P_lam = sym.lambdify([v, ds], P, 'numpy')

            ############################################################################################################
            # Constraints
            ############################################################################################################
            # Calculate Acceleration [m/s²] and Driving Force [kN]
            a = []
            F_dr = []
            for k in range(N - 1):
                a.append((v[k + 1] ** 2 - v[k] ** 2) / (2 * ds[k]))
                F_dr.append(self.Car.m * a[k] + 0.5 * self.Car.rho * self.Car.c_w * self.Car.A * v[k] ** 2)

            # Calculate Steer Angle [rad]
            delta = []
            for k in range(N):
                delta.append(sym.atan2(kappa[k] * self.Car.L, 1))

            # Calculate Time Step [s]
            dt = []
            for k in range(N - 1):
                dt.append(-v[k] / a[k] + ((v[k] / a[k]) ** 2 + 2 * ds[k] / a[k]) ** 0.5)

            # Constraints
            h = []
            lb = []
            ub = []

            # --- Boundary Conditions
            # Velocity Boundary
            for k in range(N):
                if k == N - 1 and vel_end:
                    ub.append(v_end - v[k])
                    lb.append(self.Car.v_min - v[k])
                    h.append(v[k])
                elif k == 0 and vel_start:
                    ub.append(0.0)
                    lb.append(0.0)
                    h.append(v[k])
                else:
                    lb.append(self.Car.v_min - v[k])
                    ub.append(self.Car.v_max - v[k])
                    h.append(v[k])

            # Driving Force Boundary
            for k in range(N - 1):
                lb.append(self.Car.F_br_max - F_dr[k])
                ub.append(self.Car.F_dr_max - F_dr[k])
                h.append(F_dr[k])

            # Steering Rate Boundary
            for k in range(N - 1):
                lb.append(- self.Car.v_delta_max - (delta[k + 1] - delta[k]) * v[k] / ds[k])
                ub.append(self.Car.v_delta_max - (delta[k + 1] - delta[k]) * v[k] / ds[k])
                h.append((delta[k + 1] - delta[k]) * v[k] / ds[k])

            # Acceleration Boundary
            if self.sol_options[self.key]['Friction'] == "Circle":
                for k in range(N - 1):  # Lateral Acceleration Constraint (CIRCLE)
                    lb.append(-np.inf)
                    ub.append(1 - a[k] ** 2 / ax_max[k] ** 2 - (kappa[k] * v[k] ** 2) ** 2 / ay_max[k] ** 2)
                    h.append(a[k] ** 2 / ax_max[k] ** 2 + (kappa[k] * v[k] ** 2) ** 2 / ay_max[k] ** 2)

            elif self.sol_options[self.key]['Friction'] == "Diamond":
                for k in range(N - 1):  # Lateral Acceleration Constraint (DIAMOND) (1/4)
                    lb.append(- np.inf)
                    ub.append(1 - a[k] / ax_max[k] - (kappa[k] * v[k] ** 2) / ay_max[k])
                    h.append(a[k] / ax_max[k] + (kappa[k] * v[k] ** 2) / ay_max[k])

                for k in range(N - 1):  # Lateral Acceleration Constraint (DIAMOND) (2/4)
                    lb.append(- np.inf)
                    ub.append(1 - a[k] / ax_max[k] + (kappa[k] * v[k] ** 2) / ay_max[k])
                    h.append(a[k] / ax_max[k] - (kappa[k] * v[k] ** 2) / ay_max[k])

                for k in range(N - 1):  # Lateral Acceleration Constraint (DIAMOND) (3/4)
                    lb.append(- np.inf)
                    ub.append(1 + a[k] / ax_max[k] - (kappa[k] * v[k] ** 2) / ay_max[k])
                    h.append(- a[k] / ax_max[k] + (kappa[k] * v[k] ** 2) / ay_max[k])

                for k in range(N - 1):  # Lateral Acceleration Constraint (DIAMOND) (4/4)
                    lb.append(- np.inf)
                    ub.append(1 + a[k] / ax_max[k] + (kappa[k] * v[k] ** 2) / ay_max[k])
                    h.append(- a[k] / ax_max[k] - (kappa[k] * v[k] ** 2) / ay_max[k])

            # Power Boundary
            for k in range(N - 1):
                lb.append(-np.inf)
                ub.append(P_max[k] - F_dr[k] * v[k])
                h.append(F_dr[k] * v[k])

            # Calculate Jacobian Matrix
            print('Calculate Matrix A')
            A = sym.Matrix([h]).jacobian(var)

            # Lambdify
            print('Lambdify Matrix A, lb and ub')
            A_lam = sym.lambdify([v, ds, kappa, v_end, P_max, ax_max, ay_max], A, 'numpy')
            lb_lam = sym.lambdify([v, ds, kappa, v_end, P_max, ax_max, ay_max], lb, 'numpy')
            ub_lam = sym.lambdify([v, ds, kappa, v_end, P_max, ax_max, ay_max], ub, 'numpy')
            self.P_lam = P_lam
            self.q_lam = q_lam
            self.A_lam = A_lam
            self.lb_lam = lb_lam
            self.ub_lam = ub_lam

            # Save Lambdified_Function
            print('Save Matrix to File')
            mod_local_trajectory_path = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            sys.path.append(mod_local_trajectory_path)

            if self.sol_options[self.key]['Friction'] == "Circle":
                file = "Osqp" + "_" + "kinematic_bicycle" + "_" + str(N) + ".pkl"
            elif self.sol_options[self.key]['Friction'] == "Diamond":
                file = "Osqp" + "_" + "kinematic_bicycle" + "_" + str(N) + "_" + str(
                    self.sol_options[self.key]['Friction']) + ".pkl"
            filename = mod_local_trajectory_path + '/vp_qp/opt_postproc/src/Lambdify_Function/' + file

            dill.settings['recurse'] = True
            dill.dump([P_lam, q_lam, A_lam, lb_lam, ub_lam], open(filename, "wb"))

    # Dynamic bicycle model
    def sol_init_dm(self,
                    params_path: str):
        """Function to initialize the OSQP solver with the dynamic-bicycle model \n
        by defining the objective function and constraints.\n
        Saves the matrix and vectors in order to re-use the same QP and avoid recalculations.

        :param params_path: absolute path to folder containing config file .ini

        :Authors:
            Tobias Klotz <tobias.klotz@tum.de>

        :Created on:
            16.06.2020
        """

        # Length of planning horizion
        N = self.m

        # Open lambdified functions if possible
        mod_local_trajectory_path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        sys.path.append(mod_local_trajectory_path)

        if self.sol_options[self.key]['Friction'] == "Circle":
            file = "Osqp" + "_" + "dynamic_bicycle" + "_" + str(N) + ".pkl"
        elif self.sol_options[self.key]['Friction'] == "Diamond":
            file = "Osqp" + "_" + "dynamic_bicycle" + "_" + str(N) + "_" + str(self.sol_options[self.key]['Friction']) \
                   + ".pkl"
        filename = params_path + '/Lambdify_Function/' + file

        try:
            f = open(filename)
            available = True
        except IOError:
            print("Lambda functions file not accessible! Rebuilding ...")
            available = False
        finally:
            if available:
                f.close()

        if available:
            dill.settings['recurse'] = True
            self.P_lam, self.q_lam, self.A_lam, self.lb_lam, self.ub_lam = dill.load(open(filename, "rb"))

        else:
            # Decide if Velocity at the beginning and end of the planning horizon is fixed
            vel_start = True
            vel_end = True

            # --- Define Variables ana Parameter:

            ############################################################################################################
            # Optimization Variables
            ############################################################################################################
            # velocity [m/s]
            v = sym.symbols('v0:%d' % (N))
            # Slip Angle [rad]
            beta = sym.symbols('beta0:%d' % N)
            # Gear Rate [rad/s]
            # omega = symbols('omega0:%d' % N)
            # Driving Force [kN]
            F_dr = sym.symbols('F_dr0:%d' % (N - 1))
            # Braking Force [kN]
            F_br = sym.symbols('F_br0:%d' % (N - 1))
            # Steer Angle [rad]
            delta = sym.symbols('delta0:%d' % (N - 1))
            # Optimization Vector
            var = v + beta + F_dr + F_br + delta

            ############################################################################################################
            # Online Parameters
            ############################################################################################################
            # curvature [1/m]
            kappa = sym.symbols('kappa0:%d' % (N))
            # discretization step length [m]
            ds = sym.symbols('ds:0%d' % (N - 1))
            # initial velocity [m/s]
            v_ini = sym.symbols('v_ini')
            # end velocity [m/s]
            v_end = sym.symbols('v_end')
            # max. velocity [m/s]
            v_max = sym.symbols('v_max0:%d' % (N))
            # max. power [kW]
            P_max = sym.symbols('P_max0:%d' % (N))
            # max. acceleration in x-direction of the vehicle [m/s²]
            ax_max = sym.symbols('ax_max0:%d' % (N))
            # max. acceleration in y-direction of the vehicle [m/s²]
            ay_max = sym.symbols('ay_max0:%d' % (N))

            ############################################################################################################
            # Objective function
            ############################################################################################################
            opt_func = 0
            for k in range(1, N):
                opt_func += ds[k - 1] / v[k]

            # Calculate Jacobi Matrix
            q = sym.Matrix([opt_func]).jacobian(var)
            q_lam = sym.lambdify([v, ds], q, 'numpy')

            # Calculate Hessian Matrix
            print('Calculate Matrix P')
            P = sym.hessian(opt_func, var)
            P_lam = sym.lambdify([v, ds], P, 'numpy')

            ############################################################################################################
            # Constraints
            ############################################################################################################
            h = []
            lb = []
            ub = []
            # --- Boundary Conditions/ Inequality Constraints

            # Velocity boundary condition
            for k in range(N):
                if k == N - 1 and vel_end:
                    ub.append(v_end - v[k])
                    lb.append(self.Car.v_min - v[k])
                    h.append(-v[k])
                elif k == 0 and vel_start:
                    ub.append(v_ini - v[k])
                    lb.append(v_ini - v[k])
                    h.append(-v[k])
                else:
                    lb.append(self.Car.v_min - v[k])
                    ub.append(v_max[k] - v[k])
                    h.append(-v[k])

            # Slip angle boundary condition
            for k in range(N):
                lb.append(- self.Car.beta_max - beta[k])
                ub.append(self.Car.beta_max - beta[k])
                h.append(-beta[k])

            # Driving force boundary condition
            for k in range(N - 1):
                lb.append(- F_dr[k])
                ub.append(self.Car.F_dr_max - F_dr[k])
                h.append(-F_dr[k])

            # Braking force boundary condition
            for k in range(N - 1):
                lb.append(self.Car.F_br_max - F_br[k])
                ub.append(- F_br[k])
                h.append(-F_br[k])

            # Steering rate boundary condition
            for k in range(N - 1):
                lb.append(- self.Car.delta_max - delta[k])
                ub.append(self.Car.delta_max - delta[k])
                h.append(- delta[k])

            # --- FORCES
            alpha_f = []
            alpha_r = []
            ma_x = []
            ma_y = []

            F_d = []
            F_xf = []
            F_xr = []
            F_yf = []
            F_yr = []
            F_zf = []
            F_zr = []

            # Roll Friction (front/rear) [kN]
            F_roll_f = self.Car.f_r * self.Car.m * self.Car.g * self.Car.l_r / (self.Car.l_r + self.Car.l_f)
            F_roll_r = self.Car.f_r * self.Car.m * self.Car.g * self.Car.l_f / (self.Car.l_r + self.Car.l_f)

            for k in range(N - 1):
                # tire slip angle (front & rear)
                alpha_f = np.append(alpha_f,
                                    delta[k] - sym.atan2((self.Car.l_f * kappa[k] * v[k]
                                                          / (2 * np.pi) + v[k] * sym.sin(beta[k])),
                                                         (v[k] * sym.cos(beta[k]))))
                alpha_r = np.append(alpha_r, sym.atan2((self.Car.l_r * kappa[k] * v[k]
                                                        / (2 * np.pi) - v[k] * sym.sin(beta[k])),
                                                       (v[k] * sym.cos(beta[k]))))

                # aerodynamic resistance [kN]
                F_d = np.append(F_d, 0.5 * self.Car.c_w * self.Car.rho * self.Car.A * v[k] ** 2)

                # force at axle in x-direction (front & rear)
                F_xf = np.append(F_xf, self.Car.k_dr * F_dr[k] + self.Car.k_br * F_br[k] - F_roll_f)
                F_xr = np.append(F_xr, (1 - self.Car.k_dr) * F_dr[k] + (1 - self.Car.k_br) * F_br[k] - F_roll_r)

                # total force in x-direction at CoG
                ma_x = np.append(ma_x, F_xf[k] + F_xr[k] - F_d[k])

                # force at axle in z-direction (front & rear)
                F_zf = np.append(F_zf, self.Car.m * self.Car.g * self.Car.l_r / (self.Car.l_r + self.Car.l_f)
                                 - self.Car.h_cg / (self.Car.l_r + self.Car.l_f) * ma_x[k]
                                 + 0.5 * self.Car.c_lf * self.Car.rho * self.Car.A * v[k] ** 2)
                F_zr = np.append(F_zr, self.Car.m * self.Car.g * self.Car.l_f / (self.Car.l_r + self.Car.l_f)
                                 + self.Car.h_cg / (self.Car.l_r + self.Car.l_f) * ma_x[k]
                                 + 0.5 * self.Car.c_lr * self.Car.rho * self.Car.A * v[k] ** 2)

                # force at axle in y-direction (front & rear)
                F_yf = np.append(F_yf, self.Car.D_f * (1 + self.Car.eps_f * F_zf[k] / self.Car.F_z0) * F_zf[k]
                                 / self.Car.F_z0
                                 * sym.sin(self.Car.C_f * sym.atan2(self.Car.B_f * alpha_f[k]
                                                                    - self.Car.E_f
                                                                    * (self.Car.B_f * alpha_f[k]
                                                                       - sym.atan2(self.Car.B_f * alpha_f[k], 1)), 1)))
                F_yr = np.append(F_yr, self.Car.D_r * (1 + self.Car.eps_r * F_zr[k] / self.Car.F_z0) * F_zr[k]
                                 / self.Car.F_z0 * sym.sin(self.Car.C_r * sym.atan2(self.Car.B_r * alpha_r[k]
                                                                                    - self.Car.E_r
                                                                                    * (self.Car.B_r * alpha_r[k]
                                                                                    - sym.atan2(self.Car.B_r
                                                                                                * alpha_r[k], 1)), 1)))

                # total force in y-direction at CoG
                ma_y = np.append(ma_y, F_yr[k] + F_xf[k] * sym.sin(delta[k]) + F_yf[k] * sym.cos(delta[k]))

            # --- EQUALITY CONSTRAINTS (Velocity, Slip Angle)
            for k in range(N - 1):
                h = np.append(h,
                              [  # Derivation of Velocity (Christ Eq. 5.2)
                                  v[k + 1] - v[k] - ds[k] / v[k]
                                  * (1 / self.Car.m * (+ F_xr[k] * sym.cos(beta[k])
                                                       + F_xf[k] * sym.cos(delta[k] - beta[k])
                                                       + F_yr[k] * sym.sin(beta[k])
                                                       - F_yf[k] * sym.sin(delta[k] - beta[k])
                                                       - F_d[k] * sym.cos(beta[k]))),
                                  # Derivation of Slip Angle (Christ Eq. 5.3)
                                  (beta[k + 1] - beta[k]) / (ds[k] / v[k]) - 1 / (2 * np.pi)
                                  * (-kappa[k] * v[k]
                                     + 1 / (self.Car.m * v[k])
                                     * (- F_xr[k] * sym.sin(beta[k])
                                        + F_xf[k] * sym.sin(delta[k] - beta[k])
                                        + F_yr[k] * sym.cos(beta[k])
                                        + F_yf[k] * sym.cos(delta[k] - beta[k])
                                        + F_d[k] * sym.sin(beta[k])))])
                # Lower Bound
                lb = np.append(lb,
                               [  # Derivation of Velocity (Christ Eq. 5.2)
                                   v[k + 1] - v[k] - ds[k] / v[k]
                                   * (1 / self.Car.m * (+ F_xr[k] * sym.cos(beta[k])
                                                        + F_xf[k] * sym.cos(delta[k] - beta[k])
                                                        + F_yr[k] * sym.sin(beta[k])
                                                        - F_yf[k] * sym.sin(delta[k] - beta[k])
                                                        - F_d[k] * sym.cos(beta[k]))),
                                   # Derivation of Slip Angle (Christ Eq. 5.3)
                                   (beta[k + 1] - beta[k]) / (ds[k] / v[k]) - 1 / (2 * np.pi)
                                   * (- kappa[k] * v[k]
                                      + 1 / (self.Car.m * v[k])
                                      * (- F_xr[k] * sym.sin(beta[k])
                                         + F_xf[k] * sym.sin(delta[k] - beta[k])
                                         + F_yr[k] * sym.cos(beta[k])
                                         + F_yf[k] * sym.cos(delta[k] - beta[k])
                                         + F_d[k] * sym.sin(beta[k])))])
                # Upper Bound
                ub = np.append(ub,
                               [  # Derivation of Velocity (Christ Eq. 5.2)
                                   v[k + 1] - v[k] - ds[k] / v[k]
                                   * (1 / self.Car.m * (+ F_xr[k] * sym.cos(beta[k])
                                                        + F_xf[k] * sym.cos(delta[k] - beta[k])
                                                        + F_yr[k] * sym.sin(beta[k])
                                                        - F_yf[k] * sym.sin(delta[k] - beta[k])
                                                        - F_d[k] * sym.cos(beta[k]))),
                                   # Derivation of Slip Angle (Christ Eq. 5.3)
                                   (beta[k + 1] - beta[k]) / (ds[k] / v[k]) - 1 / (2 * np.pi)
                                   * (- kappa[k] * v[k] + 1 / (self.Car.m * v[k])
                                      * (- F_xr[k] * sym.sin(beta[k]) + F_xf[k] * sym.sin(delta[k] - beta[k]) + F_yr[k]
                                         * sym.cos(beta[k]) + F_yf[k] * sym.cos(delta[k] - beta[k]) + F_d[k]
                                         * sym.sin(beta[k])))])

            # INEQUALITY CONSTRAINTS
            # Friction Coefficient
            mu_x = []
            mu_y = []
            for k in range(N - 1):
                mu_x.append(ax_max[k] / self.Car.a_max)
                mu_y.append(ay_max[k] / self.Car.a_lat_max)

            for k in range(N - 1):
                if k == N - 2:
                    h = np.append(h,
                                  [  # Power Constraint
                                      P_max[k] - v[k] * F_dr[k],
                                      # Braking and Driving Force Constraint
                                      - F_dr[k] * F_br[k],
                                      # Kamm Circle Front Axle
                                      - (F_xf[k] / (mu_x[k] * F_zf[k])) ** 2 - (F_yf[k] / (mu_y[k] * F_zf[k])) ** 2,
                                      # Kamm Circle Rear Axle
                                      - (F_xr[k] / (mu_x[k] * F_zr[k])) ** 2 - (F_yr[k] / (mu_y[k] * F_zr[k])) ** 2])
                    # Lower Bound
                    lb = np.append(lb,
                                   [  # Power Constraint
                                       - v[k] * F_dr[k],
                                       # Braking and Driving Force Constraint
                                       - 0.02 - F_dr[k] * F_br[k],
                                       # Kamm Circle Front Axle
                                       - (F_xf[k] / (mu_x[k] * F_zf[k])) ** 2 - (F_yf[k] / (mu_y[k] * F_zf[k])) ** 2,
                                       # Kamm Circle Rear Axle
                                       - (F_xr[k] / (mu_x[k] * F_zr[k])) ** 2 - (F_yr[k] / (mu_y[k] * F_zr[k])) ** 2])
                    # Upper Bound
                    ub = np.append(ub,
                                   [  # Power Constraint
                                       P_max[k] - v[k] * F_dr[k],
                                       # Braking and Driving Force Constraint
                                       - F_dr[k] * F_br[k],
                                       # Kamm Circle Front Axle
                                       1.0 - (F_xf[k]
                                              / (mu_x[k] * F_zf[k])) ** 2 - (F_yf[k] / (mu_y[k] * F_zf[k])) ** 2,
                                       # Kamm Circle Rear Axle
                                       1.0 - (F_xr[k]
                                              / (mu_x[k] * F_zr[k])) ** 2 - (F_yr[k] / (mu_y[k] * F_zr[k])) ** 2])
                else:
                    h = np.append(h,
                                  [  # Power Constraint
                                      P_max[k] - v[k] * F_dr[k],
                                      # Braking and Driving Force Constraint
                                      - F_dr[k] * F_br[k],
                                      # Constraint Derivative of Driving Force
                                      - ((F_dr[k + 1] - F_dr[k]) / (ds[k] / v[k])),
                                      # Constraint Derivative of Braking Force
                                      - ((F_br[k + 1] - F_br[k]) / (ds[k] / v[k])),
                                      # Constraint Derivative of Steer Angle
                                      - ((delta[k + 1] - delta[k]) / (ds[k] / v[k])),
                                      # Kamm Circle Front Axle
                                      - (F_xf[k] / (mu_x[k] * F_zf[k])) ** 2 - (F_yf[k] / (mu_y[k] * F_zf[k])) ** 2,
                                      # Kamm Circle Rear Axle
                                      - (F_xr[k] / (mu_x[k] * F_zr[k])) ** 2 - (F_yr[k] / (mu_y[k] * F_zr[k])) ** 2])
                    # Lower Bound
                    lb = np.append(lb,
                                   [  # Power Constraint
                                       - v[k] * F_dr[k],
                                       # Braking and Driving Force Constraint
                                       - 0.02 - F_dr[k] * F_br[k],
                                       # Constraint Derivative of Driving Force
                                       - np.inf,
                                       # Constraint Derivative of Braking Force
                                       self.Car.F_br_max - ((F_br[k + 1] - F_br[k]) / (ds[k] / v[k])),
                                       # Constraint Derivative of Steer Angle
                                       - self.Car.delta_max - ((delta[k + 1] - delta[k]) / (ds[k] / v[k])),
                                       # Kamm Circle Front Axle
                                       - (F_xf[k] / (mu_x[k] * F_zf[k])) ** 2 - (F_yf[k] / (mu_y[k] * F_zf[k])) ** 2,
                                       # Kamm Circle Rear Axle
                                       - (F_xr[k] / (mu_x[k] * F_zr[k])) ** 2 - (F_yr[k] / (mu_y[k] * F_zr[k])) ** 2])
                    # Upper Bound
                    ub = np.append(ub,
                                   [  # Power Constraint
                                       P_max[k] - v[k] * F_dr[k],
                                       # Braking and Driving Force Constraint
                                       - F_dr[k] * F_br[k],
                                       # Constraint Derivative of Driving Force
                                       self.Car.F_dr_max - ((F_dr[k + 1] - F_dr[k]) / (ds[k] / v[k])),
                                       # Constraint Derivative of Braking Force
                                       np.inf,
                                       # Constraint Derivative of Steer Angle
                                       self.Car.delta_max - ((delta[k + 1] - delta[k]) / (ds[k] / v[k])),
                                       # Kamm Circle Front Axle
                                       1.0 - (F_xf[k] / (mu_x[k] * F_zf[k])) ** 2
                                       - (F_yf[k] / (mu_y[k] * F_zf[k])) ** 2,
                                       # Kamm Circle Rear Axle
                                       1.0 - (F_xr[k] / (mu_x[k] * F_zr[k])) ** 2
                                       - (F_yr[k] / (mu_y[k] * F_zr[k])) ** 2])

            # Calculate Jacobian Matrix
            print('Caluclate Matrix A')
            A = sym.Matrix([-h]).jacobian(var)
            print('Matrix A solved.')
            print('Lambdify Matrix')
            A_lam = sym.lambdify([v, beta, F_dr, F_br, delta, ds, kappa, v_ini, v_end, v_max, P_max, ax_max, ay_max], A,
                                 'numpy')
            print('a_lam lambdified.')
            lb_lam = sym.lambdify([v, beta, F_dr, F_br, delta, ds, kappa, v_ini, v_end, v_max, P_max, ax_max, ay_max],
                                  lb, modules=['numpy', 'math'])
            print('lbo_lam lambdified.')
            ub_lam = sym.lambdify([v, beta, F_dr, F_br, delta, ds, kappa, v_ini, v_end, v_max, P_max, ax_max, ay_max],
                                  ub, modules=['numpy', 'math'])
            print('lbo lambdified.')

            self.P_lam = P_lam
            self.q_lam = q_lam
            self.A_lam = A_lam
            self.lb_lam = lb_lam
            self.ub_lam = ub_lam

            # Save Lambdified_Function
            print('Save Lambdified functions to File')

            if self.sol_options[self.key]['Friction'] == "Circle":
                file = "Osqp" + "_" + "dynamic_bicycle" + "_" + str(N) + ".pkl"
            elif self.sol_options[self.key]['Friction'] == "Diamond":
                file = "Osqp" + "_" + "dynamic_bicycle" + "_" + str(N) + "_" + str(
                    self.sol_options[self.key]['Friction']) + ".pkl"

            filepath = params_path + '/Lambdify_Function/'
            filename = filepath + file
            os.makedirs(filepath, exist_ok=True)

            dill.settings['recurse'] = True
            dill.dump([P_lam, q_lam, A_lam, lb_lam, ub_lam], open(filename, "wb"))

    def calc_v_osqp(self,
                    N: int,
                    ds: np.ndarray,
                    kappa: np.ndarray,
                    P_max: float = None,
                    ax_max: np.array = None,
                    ay_max: np.array = None,
                    x0_v: np.ndarray = None,
                    v_max: np.ndarray = None,
                    v_end: float = None,
                    F_ini: float = None
                    ):
        """Function to update the matrix and functions for the optimization problem. \n
        Solve the optimization problem with the solver OSQP.

        :param N: number of velocity points
        :param ds: discretization step length of given path [m]
        :param kappa: curvature profile of given path [rad/m]
        :param P_max: max. allowed power [kW]
        :param ax_max: max. allowed longitudinal acceleration [m/s^2]
        :param ay_max: max. allowed lateral accelereation [m/s]
        :param x0_v: initial guess velocity [m/s]
        :param v_max: max. allowed velocity (in objective function) [m/s]
        :param v_end: hard constrained max. allowed value of end velocity in optimization horizon [m/s]
        :param F_ini: hard constrained initial force [kN]

        :return: v: v: optimized velocity [m/s] \n
            F: optimize powertrain force [kN] \n
            P: optimized power force [kW] \n
            ax: acceleration in x-direction of CoG [m/s²] \n
            ay: acceleration in y-direction of CoG [m/s²]
            t_total: runtime of the solver OSQP [ms] \n
            res.info.status: status of the solution (solved, infeasible, etc.)

        :Authors:
            Tobias Klotz <tobias.klotz@tum.de>

        :Created on:
            16.06.2020
        """

        # Initialisation of Optimization with the Point mass model
        if self.sol_options[self.key]['Model'] == "PM":
            # Initialize Velocity
            if not v_end:
                v_end = self.Car.v_end
            if len(x0_v) == 0:
                v = np.ones(N)
            else:
                v = x0_v
                if x0_v[0] == 0:
                    v[0] = 0.1
                else:
                    v[0] = x0_v[0]

            # Initialize Velocity k=0
            v_ini = x0_v[0]

            # Initialize Force k=0
            if F_ini == []:
                F_ini = 0.0

            # Initialize max. Power
            if self.sol_options[self.key]['VarPower']:
                pass
            else:
                P_max = self.Car.P_max * np.ones(N - 1)
            # Initialize max. velocity
            if v_max == []:
                v_max = self.Car.v_max * np.ones(N)
            # Initialize max. acceleration
            if ax_max is None:
                ax_max = self.Car.a_max * np.ones(N)
            if ay_max is None:
                ay_max = self.Car.a_lat_max * np.ones(N)

            '''# Choose step length
            if self.sol_options[self.key]['Friction'] == "Diamond":
                if self.sol_options[self.key]['VarPower']:
                    alpha = 0.2
                else:
                    alpha = 1
            elif self.sol_options[self.key]['Friction'] == "Circle":
                alpha = 0.4
            else:
                alpha = 1'''
            # Choose step length
            alpha = self.sol_options[self.key]['Alpha']

        # Kineamtic bicycle model
        elif self.sol_options[self.key]['Model'] == "KM":
            # Initialize Velocity
            if len(x0_v) == 0:
                v = v_end * np.ones(N)
                v[0] = 0.1
            else:
                v = x0_v
                if x0_v[0] == 0:
                    v[0] = 0.1
            # Initialize Force
            if F_ini == []:
                F_ini = 0.0
            # Initialize max. Power
            if self.sol_options[self.key]['VarPower']:
                pass
            else:
                P_max = self.Car.P_max * np.ones(N - 1)
            # Initialize max. velocity
            if v_max == []:
                v_max = self.Car.v_max * np.ones(N)
            # Initialize max. acceleartion
            if ax_max is None:
                ax_max = self.Car.a_max * np.ones(N)
            if ay_max is None:
                ay_max = self.Car.a_lat_max * np.ones(N)
            # Initialize end velocity
            if v_end is None:
                v_end = self.Car.v_end

            # Choose step length
            alpha = self.sol_options[self.key]['Alpha']

        # Dynamic bicycle model
        elif self.sol_options[self.key]['Model'] == "DM":
            # Initialize end velocity
            if not v_end:
                v_end = self.Car.v_end
            # Initialize start velocity
            if x0_v[0] == 0:
                v_ini = 0.1
            # Initialize velocity
            else:
                v_ini = x0_v[0]
            v = v_end * np.ones(N)
            if x0_v[0] == 0:
                v[0] = 0.1

            # Initialize Force
            if F_ini == []:
                F_ini = 0.0
            # Initialize max. power
            if self.sol_options[self.key]['VarPower']:
                pass
            else:
                P_max = self.Car.P_max * np.ones(N - 1)
            # Initialize max. velocity
            if v_max == []:
                v_max = self.Car.v_max * np.ones(N)
            # Initialize max. acceleartion
            if ax_max is None:
                ax_max = self.Car.a_max * np.ones(N)
            if ay_max is None:
                ay_max = self.Car.a_lat_max * np.ones(N)

            # Initialize Optimization variables
            beta = np.zeros(N)  # slip angle
            F_dr = np.zeros(N - 1)  # driving force
            F_br = np.zeros(N - 1)  # braking force
            if F_ini > 0:
                F_dr[0] = F_ini
            else:
                F_br[0] = F_ini
            delta = np.zeros(N - 1)  # steer angle

            kappa = kappa[0:N]  # curvature

            # Choose step length
            alpha = self.sol_options[self.key]['Alpha']  # alpha = 0.1

        # While-Loop to Solve the QPs
        count = True    # maximum number iterations reached
        infeasible = False    # solution is infeasible
        t = []  # save calculation time
        save_obj = []   # save objective
        counter = 0     # count the iterations
        v_start = 0.1
        while count:
            # Fill Matrix P, q, A, b, G, h
            if self.sol_options[self.key]['Model'] == "PM":
                P = sparse.csc_matrix(self.P_lam(v, ds))
                # pprint(P)
                q = np.array(self.q_lam(v, ds).T)
                # pprint(q)
                A = sparse.csc_matrix(self.A_lam(v, F_ini, ds, kappa, v_ini, v_end, v_max, P_max, ax_max, ay_max))
                # pprint(A)
                lbo = np.array(self.lb_lam(v, F_ini, ds, kappa, v_ini, v_end, v_max, P_max, ax_max, ay_max))
                # pprint(lbo)
                ubo = np.array(self.ub_lam(v, F_ini, ds, kappa, v_ini, v_end, v_max, P_max, ax_max, ay_max))
                # pprint(ubo)
            elif self.sol_options[self.key]['Model'] == "KM":
                P = sparse.csc_matrix(self.P_lam(v, ds))
                # pprint(P)
                q = np.array(self.q_lam(v, ds).T)
                # pprint(q)
                A = sparse.csc_matrix(self.A_lam(v, ds, kappa, v_end, P_max, ax_max, ay_max))
                # pprint(A)
                lbo = np.array(self.lb_lam(v, ds, kappa, v_end, P_max, ax_max, ay_max))
                # pprint(lbo)
                ubo = np.array(self.ub_lam(v, ds, kappa, v_end, P_max, ax_max, ay_max))
                # pprint(ubo)
            elif self.sol_options[self.key]['Model'] == "DM":
                P = sparse.csc_matrix(self.P_lam(v, ds))
                # pprint(P)
                q = np.array(self.q_lam(v, ds).T)
                # pprint(q)
                A = sparse.csc_matrix(self.A_lam(v, beta, F_dr, F_br, delta, ds, kappa, v_ini, v_end, v_max, P_max,
                                                 ax_max, ay_max))
                # pprint(A)
                lbo = np.array(self.lb_lam(v, beta, F_dr, F_br, delta, ds, kappa, v_ini, v_end, v_max, P_max, ax_max,
                                           ay_max))
                # pprint(lbo)
                ubo = np.array(self.ub_lam(v, beta, F_dr, F_br, delta, ds, kappa, v_ini, v_end, v_max, P_max, ax_max,
                                           ay_max))

            # Start timer
            t0 = time.perf_counter()

            # Create an OSQP object
            prob = osqp.OSQP()

            # Setup workspace
            prob.setup(P, q, A, lbo, ubo, warm_start=True, verbose=0)

            # Solve
            res = prob.solve()

            # Stop Timer
            t1 = time.perf_counter()

            # save Calculation Time
            t.append(t1 - t0)
            print(res.info.status)

            if counter == 30:  # stop while-loop due to max. iteartions reached
                t_total = sum(t)
                count = False
            else:
                counter += 1

            # Check if solution could be found
            if res.info.status == 'maximum iterations reached':
                t_total = sum(t)
                # count = False
                print('No solution')

            # Adapt Velocity, Acceleration, Delta and Time with new values
            if res.info.status == 'primal infeasible':
                # Try restart if no solution could be found
                if self.sol_options[self.key]['Model'] == "PM" or self.sol_options[self.key]['Model'] == "KM":
                    if abs(res.info.obj_val) < 0.1:
                        pass
                    else:
                        if infeasible:
                            print('Try new start')
                            v = v_end * np.ones(N)
                            v_start = 0.9 * v_start
                            v[0] = v_start
                            F_ini = 0.0
                            alpha = 1

                        else:
                            print('Try new start')
                            v = v_end * np.ones(N)
                            v_start = 0.9 * x0_v[0]
                            v[0] = v_start
                            # F_ini = 0.0
                            alpha = 0.1
                            infeasible = True

            else:
                # Adapt optimaization variables with solution
                if self.sol_options[self.key]['Model'] == "PM":
                    v += alpha * res.x[0:N]  # v(N): Velocity

                elif self.sol_options[self.key]['Model'] == "KM":
                    v += alpha * res.x[0:N]  # v(N): Velocity

                elif self.sol_options[self.key]['Model'] == "DM":
                    if counter == 2:
                        alpha = 0.2
                    v += alpha * res.x[0:N]  # v(N): Velocity
                    beta += alpha * res.x[N:2 * N]  # beta(N): Slip Angle
                    F_dr += alpha * res.x[2 * N:3 * N - 1]  # F_dr(N-1): Driving Force
                    F_br += alpha * res.x[3 * N - 1:4 * N - 2]  # F_br(N-1): Braking Force
                    delta += alpha * res.x[4 * N - 2:5 * N - 3]  # delta(N-1): Steering Angle

                # Check objective if it is below the tolerance
                if not len(save_obj) == 0 and self.sol_options[self.key]['Model'] == "DM":
                    # adaptive alpha control for dynamic bicylcle model
                    if abs(res.info.obj_val) < 0.001:
                        t_total = sum(t)
                        count = False
                        pass
                    elif abs(res.info.obj_val) < 0.01:
                        alpha = 1
                        t_total = sum(t)
                        count = False
                    elif abs(res.info.obj_val) < 1:
                        alpha = 0.5
                    elif abs(res.info.obj_val) < 3:
                        alpha = 0.4

                elif not len(save_obj) == 0:
                    # adaptive alpha control for friction circel with ppint mass model and kineamtic bicylce model
                    if abs(res.info.obj_val) < 2 and self.sol_options[self.key]['Friction'] == "Circle":
                        alpha = 0.2
                    if abs(res.info.obj_val) < 0.2 and (self.sol_options[self.key]['Friction'] == "Circle"
                                                        ):
                        alpha = 0.2
                    if abs(res.info.obj_val) < 0.01:
                        t_total = sum(t)
                        count = False

                save_obj = np.append(save_obj, res.info.obj_val)

        # Return solution, velocity and result
        if res.info.status == 'primal infeasible':
            v = np.zeros(N)
            F = np.zeros(N - 1)
            P = np.zeros(N - 1)
            ax = np.zeros(N - 1)
            ay = np.zeros(N - 1)
        else:
            if self.sol_options[self.key]['Model'] == "PM":
                sol = v

            if self.sol_options[self.key]['Model'] == "KM":
                sol = v
            if self.sol_options[self.key]['Model'] == "DM":
                sol = []
                sol.append(v)
                sol.append(beta)
                sol.append(F_dr)
                sol.append(F_br)
                sol.append(delta)
                sol = np.concatenate(sol)

            v, F, P, ax, ay = self.transform_results(sol, ds, kappa, N)

        return v, F, P, ax, ay, t_total, res.info.status

    def transform_results(self, sol, ds, kappa, N):
        """Function to re-calculate the optimization variables of the QP.

        :param sol: solution of the QP
        :param ds: discretization step length of given path [m]
        :param kappa: curvature profile of given path [rad/m]
        :param N: number of velocity points

        :return: v: optimized velocity [m/s] \n
            F: optimize powertrain force [kN] \n
            P: optimized power force [kW] \n
            ax: acceleration in x-direction of CoG [m/s²] \n
            ay: acceleration in y-direction of CoG [m/s²]

        :Authors:
            Tobias Klotz <tobias.klotz@tum.de>

        :Created on:
            16.06.2020
        """

        if self.sol_options[self.key]['Model'] == "PM":
            v = sol

            # Calculate Acceleration
            ax = (v[1:self.m] ** 2 - v[0:self.m - 1] ** 2) / (2 * ds[0:self.m - 1])
            ay = kappa[0:self.m - 1] * v[0:self.m - 1] ** 2

            # Calculate Force
            F = self.sol_options[self.key]['Create_Solver'].Car.m * ax - \
                self.sol_options[self.key]['Create_Solver'].Car.q * 0.001 * v[1:self.m] ** 2

            # Calculate Power
            P = F * v[0:-1]

        elif self.sol_options[self.key]['Model'] == "KM":
            v = sol

            # Calculate Acceleration
            ax = (v[1:self.m] ** 2 - v[0:self.m - 1] ** 2) / (2 * ds[0:self.m - 1])
            ay = kappa[0:self.m - 1] * v[0:self.m - 1] ** 2

            # Calculate Force
            F = self.sol_options[self.key]['Create_Solver'].Car.m * ax - \
                self.sol_options[self.key]['Create_Solver'].Car.q * 0.001 * v[1:self.m] ** 2

            # Calculate Power
            P = F * v[0:-1]

        elif self.sol_options[self.key]['Model'] == "DM":
            v = sol[0:N]
            beta = sol[N:2 * N]
            F_dr = sol[2 * N:3 * N - 1]
            F_br = sol[3 * N - 1:4 * N - 2]
            delta = sol[4 * N - 2:5 * N - 3]

            # --- FORCES
            alpha_f = []
            alpha_r = []
            ma_x = []
            ma_y = []

            F_d = []
            F_xf = []
            F_xr = []
            F_yf = []
            F_yr = []
            F_zf = []
            F_zr = []

            # Roll Friction (front/rear) [kN]
            F_roll_f = self.Car.f_r * self.Car.m * self.Car.g * self.Car.l_r / (self.Car.l_r + self.Car.l_f)
            F_roll_r = self.Car.f_r * self.Car.m * self.Car.g * self.Car.l_f / (self.Car.l_r + self.Car.l_f)

            for k in range(N - 1):
                # tire slip angle (front & rear)
                alpha_f = np.append(alpha_f,
                                    delta[k] - sym.atan2(
                                        (self.Car.l_f * kappa[k] * v[k] / (2 * np.pi) + v[k] * sym.sin(beta[k])),
                                        (v[k] * sym.cos(beta[k]))))
                alpha_r = np.append(alpha_r, sym.atan2((self.Car.l_r * kappa[k]
                                                        * v[k] / (2 * np.pi) - v[k] * sym.sin(beta[k])),
                                                       (v[k] * sym.cos(beta[k]))))

                # aerodynamic resistance [kN]
                F_d = np.append(F_d, 0.5 * self.Car.c_w * self.Car.rho * self.Car.A * v[k] ** 2)

                # force at axle in x-direction (front & rear)
                F_xf = np.append(F_xf, self.Car.k_dr * F_dr[k] + self.Car.k_br * F_br[k] - F_roll_f)
                F_xr = np.append(F_xr, (1 - self.Car.k_dr) * F_dr[k] + (1 - self.Car.k_br) * F_br[k] - F_roll_r)

                # total force in x-direction at CoG
                ma_x = np.append(ma_x, F_xf[k] + F_xr[k] - F_d[k])
                ax = ma_x / self.Car.m

                # force at axle in z-direction (front & rear)
                F_zf = np.append(F_zf, self.Car.m * self.Car.g * self.Car.l_r
                                 / (self.Car.l_r + self.Car.l_f) - self.Car.h_cg
                                 / (self.Car.l_r + self.Car.l_f) * ma_x[k]
                                 + 0.5 * self.Car.c_lf * self.Car.rho * self.Car.A * v[k] ** 2)
                F_zr = np.append(F_zr, self.Car.m * self.Car.g * self.Car.l_f
                                 / (self.Car.l_r + self.Car.l_f) + self.Car.h_cg
                                 / (self.Car.l_r + self.Car.l_f) * ma_x[k]
                                 + 0.5 * self.Car.c_lr * self.Car.rho * self.Car.A * v[k] ** 2)

                # force at axle in y-direction (front & rear)
                F_yf = np.append(F_yf, self.Car.D_f * (1 + self.Car.eps_f * F_zf[k] / self.Car.F_z0) * F_zf[k]
                                 / self.Car.F_z0
                                 * sym.sin(self.Car.C_f
                                           * sym.atan2(self.Car.B_f * alpha_f[k] - self.Car.E_f
                                                       * (self.Car.B_f * alpha_f[k]
                                                          - sym.atan2(self.Car.B_f * alpha_f[k], 1)), 1)))
                F_yr = np.append(F_yr, self.Car.D_r * (1 + self.Car.eps_r * F_zr[k] / self.Car.F_z0) * F_zr[k]
                                 / self.Car.F_z0
                                 * sym.sin(self.Car.C_r
                                           * sym.atan2(self.Car.B_r * alpha_r[k] - self.Car.E_r
                                                       * (self.Car.B_r * alpha_r[k]
                                                          - sym.atan2(self.Car.B_r * alpha_r[k], 1)), 1)))

                # total force in y-direction at CoG
                ma_y = np.append(ma_y, F_yr[k] + F_xf[k] * sym.sin(delta[k]) + F_yf[k] * sym.cos(delta[k]))
                ay = ma_y / self.Car.m

                F = F_dr + F_br
                P = F * v[0:N - 1]

            do_plot = False
            if do_plot is True:
                # TUM Color
                TUMBlue = [0 / 255, 101 / 255, 189 / 255]
                TUMGray1 = [51 / 255, 51 / 255, 51 / 255]
                TUMOrange = [227 / 255, 114 / 255, 34 / 255]

                # Line Width
                LW = 1.5
                # Plot Constraints
                # FRONT TIRE
                plt.figure()
                ax1 = plt.gca()
                # Boundary
                circle = plt.Circle((0, 0), 1, facecolor='none',
                                    edgecolor=TUMGray1, linewidth=LW, alpha=0.5)
                legend = []
                ax1.add_patch(circle)
                p_a1, = plt.plot(F_yf / F_zf, F_xf / F_zf,
                                 color=TUMBlue, linewidth=LW, linestyle=':', marker='o')
                legend.append(r'$Vorderreifen_\mathrm{o,dESM}$')

                p_a1, = plt.plot(F_yr / F_zr, F_xr / F_zr,
                                 color=TUMOrange, linewidth=LW, linestyle=':', marker='o')
                legend.append(r'$Hinterreifen_\mathrm{o,dESM}$')

                plt.xlim(-1.0, 1.0)
                plt.ylim(-1.0, 1.0)
                plt.legend(legend, loc='upper right', bbox_to_anchor=(0.99, 0.95))

                # Achsenbeschriftung
                plt.xlabel(r'$F_\mathrm{y,f/r}/F_\mathrm{z,f/r}$')
                plt.ylabel(r'$F_\mathrm{x,f/r}/F_\mathrm{z,f/r}$')

                plt.show()

        return v, F, P, ax, ay


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
