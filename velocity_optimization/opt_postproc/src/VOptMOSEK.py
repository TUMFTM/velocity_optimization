try:
    import mosek
except ImportError:
    print('Warning: No module mosek found. Not necessary on car but for development.')
try:
    from sympy import symbols, Matrix, lambdify, hessian, cos, sin, atan2, SparseMatrix
except ImportError:
    print('Warning: No module sympy found. Not necessary on car but for development.')
try:
    import dill
except ImportError:
    print('Warning: No module dill found. Not necessary on car but for development.')
import numpy as np
import os
import sys
import configparser
import time
# flake8: noqa


class VOptMOSEK:

    def __init__(self,
                 m: int,
                 sid: str,
                 params_path: str,
                 vis_options: dict,
                 sol_options: dict,
                 key: str):
        """Class to optimize a velocity profile for a given path using the solver MOSEK.

        .. math::
            \min_x \qquad 1/2~x^T H_m x + q^T_v x \n
            \mathrm{s.t} \qquad blc \leq A x \leq buc \n
            blx \leq x \leq bux

        More information to the MOSEK Solver can be found at https://docs.mosek.com/9.2/pythonapi/index.html

        :param m: number of velocity points
        :param sid: optimized ID 'PerfSQP' or 'EmergSQP'
        :param params_path: absolute path to folder containing config file .ini
        :param vis_options: user specified visualization options of the debugging tool
        :param sol_options: user specified solver options of the debugging tool
        :param key: key of the used solver

        :Authors:
            Tobias Klotz <tobias.klotz@tum.de>

        :Created on:
            16.06.2020
        """

        self.m = m
        self.sid = sid
        self.F_ini_mosek = []
        self.visOptions = vis_options
        self.sol_options = sol_options
        self.key = key

        self.Car = Car(params_path=params_path)

        # Select Vehicle Dynamic Model
        if self.sol_options[self.key]['Model'] == "PM":
            self.sol_init_pm(params_path=params_path)
        if self.sol_options[self.key]['Model'] == "KM":
            self.sol_init_km(params_path=params_path)
        if self.sol_options[self.key]['Model'] == "DM":
            self.sol_init_dm(params_path=params_path)

    # Point-mass model
    def sol_init_pm(self,
                    params_path: str):
        """Function to initialize the MOSEK solver with the point-mass model \n
        by defining the objective function and constraints.\n
        Saves the matrix and vectors in order to re-use the same QP and avoid recalculations.

        :param params_path: absolute path to folder containing config file .ini

        :Authors:
            Tobias Klotz <tobias.klotz@tum.de>

        :Created on:
            16.06.2020
        """

        # Planing horizon
        N = self.m

        # Open lambdified Functions
        if self.sol_options[self.key]['Friction'] == "Circle":
            file = "Mosek" + "_" + "point_mass" + "_" + str(N) + ".pkl"
        elif self.sol_options[self.key]['Friction'] == "Diamond":
            file = "Mosek" + "_" + "point_mass" + "_" + str(N) + "_" \
                   + str(self.sol_options[self.key]['Friction']) + ".pkl"
        else:
            print("No valid friction model specified!")
            file = None

        filepath = params_path + '/Lambdify_Function/'
        filename = filepath + file

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
            self.cval_lam, self.csubj, self.qval_lam, self.qsubi, self.qsubj, self.aval_lam, self.asubi, self.asubj, \
                self.blc_lam, self.buc_lam, self.bkc, self.blx_lam, self.bux_lam, self.bkx \
                = dill.load(open(filename, "rb"))

        else:
            inf = 0.0
            ############################################################################################################
            # Optimization Variables
            ############################################################################################################
            # velocity [m/s]
            v = symbols('v0:%d' % N)
            # Optimization Vector
            var = v

            ############################################################################################################
            # Online Parameters
            ############################################################################################################
            # curvature [1/m]
            kappa = symbols('kappa0:%d' % (N))
            # discretization step length [m]
            ds = symbols('ds:0%d' % (N - 1))
            # initial velocity [m/s]
            v_ini = symbols('v_ini')
            # end velocity [m/s]
            v_end = symbols('v_end')
            # initial force [kN]
            F_ini_param = symbols('F_ini')
            # max. power [kW]
            P_max = symbols('P_max0:%d' % (N))
            # max. acceleration in x-direction of the vehicle [m/s²]
            ax_max = symbols('ax_max0:%d' % (N))
            # max. acceleration in y-direction of the vehicle [m/s²]
            ay_max = symbols('ay_max0:%d' % (N))

            ############################################################################################################
            # Objective function
            ############################################################################################################

            # Cost Function: min s_i/v_i+1
            cost_func = 0

            for k in range(N - 1):
                cost_func += ds[k] / v[k + 1]

            # Calculate the Jacobi Function
            print('Calculation of the Jacobi Matrix')
            c = SparseMatrix(Matrix([cost_func]).jacobian(var))
            cval = []
            csubj = []
            for k in range(c.RL.__len__()):
                csubj.append(c.RL[k][1])
                cval.append(c.RL[k][2])

            cval_lam = lambdify([v, ds], cval, 'numpy')
            self.cval_lam = cval_lam
            self.csubj = csubj
            print('Calculation of the Hessian Matrix')

            q = SparseMatrix(hessian(cost_func, var))
            qval = []
            qsubi = []
            qsubj = []
            for k in range(q.RL.__len__()):
                qsubi.append(q.RL[k][0])
                qsubj.append(q.RL[k][1])
                qval.append(q.RL[k][2])

            qval_lam = lambdify([v, ds], qval, 'numpy')
            self.qval_lam = qval_lam
            self.qsubi = qsubi
            self.qsubj = qsubj

            ############################################################################################################
            # Constraints
            ############################################################################################################

            # Calculate Acceleration and Driving Force
            a = []
            F_dr = []
            for k in range(N - 1):
                a.append((v[k + 1] ** 2 - v[k] ** 2) / (2 * ds[k]))
                F_dr.append(self.Car.m * a[k] + 0.5 * self.Car.rho * self.Car.c_w * self.Car.A * v[k] ** 2)

            # BOUNDARY CONDITIONS blx <= x <= buc
            bkx = []
            blx = []
            bux = []
            # Velocity boundary condition
            for k in range(len(v)):
                if k == 0:
                    bkx.append(mosek.boundkey.fx)
                    blx.append(-v[k] + v_ini)
                    bux.append(-v[k] + v_ini)
                elif k == N - 1:
                    bkx.append(mosek.boundkey.ra)
                    blx.append(0.0 - v[k])
                    bux.append(v_end - v[k])
                else:
                    bkx.append(mosek.boundkey.ra)
                    blx.append(0.0 - v[k])
                    bux.append(self.Car.v_max - v[k])

            self.bkx = bkx

            # CONSTRAINTS blc <= Ax <= buc
            print('Calculation of the h Vector')
            h = []
            bkc = []
            blc = []
            buc = []

            # INEQUALITY CONSTRAINTS
            for k in range(N - 1):

                if self.sol_options[self.key]['Friction'] == "Circle":
                    # Kamm Circle
                    h = np.append(h, [a[k] ** 2 + (kappa[k] * v[k] ** 2) ** 2,
                                      ])
                    bkc.append(mosek.boundkey.up)
                    blc.append(- np.inf)
                    buc.append(self.Car.a_lat_max ** 2 - a[k] ** 2 - (kappa[k] * v[k] ** 2) ** 2)

                elif self.sol_options[self.key]['Friction'] == "Diamond":
                    # Friction Diamond
                    h = np.append(h, [a[k] / ax_max[k] + (kappa[k] * v[k] ** 2) / ay_max[k],
                                      a[k] / ax_max[k] - (kappa[k] * v[k] ** 2) / ay_max[k],
                                      - a[k] / ax_max[k] + (kappa[k] * v[k] ** 2) / ay_max[k],
                                      - a[k] / ax_max[k] - (kappa[k] * v[k] ** 2) / ay_max[k],
                                      ])
                    bkc.append(mosek.boundkey.up)
                    blc.append(- np.inf)
                    buc.append(1 - a[k] / ax_max[k] - (kappa[k] * v[k] ** 2) / ay_max[k])

                    bkc.append(mosek.boundkey.up)
                    blc.append(- np.inf)
                    buc.append(1 - a[k] / ax_max[k] + (kappa[k] * v[k] ** 2) / ay_max[k])

                    bkc.append(mosek.boundkey.up)
                    blc.append(- np.inf)
                    buc.append(1 + a[k] / ax_max[k] - (kappa[k] * v[k] ** 2) / ay_max[k])

                    bkc.append(mosek.boundkey.up)
                    blc.append(- np.inf)
                    buc.append(1 + a[k] / ax_max[k] + (kappa[k] * v[k] ** 2) / ay_max[k])

            # Power Constraint
            for k in range(N - 1):
                h = np.append(h, [F_dr[k] * v[k]])
                bkc.append(mosek.boundkey.up)
                blc.append(-inf)
                buc.append(P_max[k] - F_dr[k] * v[k])

            # Force Constraint
            for k in range(N - 1):
                h = np.append(h, [F_dr[k]])
                if k == 0:
                    bkc.append(mosek.boundkey.fx)
                    blc.append(F_ini_param - F_dr[k])
                    buc.append(F_ini_param - F_dr[k])

                else:
                    bkc.append(mosek.boundkey.ra)
                    blc.append(self.Car.F_br_max - F_dr[k])
                    buc.append(self.Car.F_dr_max - F_dr[k])

            # Force Derivative Constraint
            for k in range(N - 2):
                h = np.append(h, F_dr[k + 1] - F_dr[k])
                bkc.append(mosek.boundkey.ra)
                blc.append(- 4 - (F_dr[k + 1] - F_dr[k]))
                buc.append(4 - (F_dr[k + 1] - F_dr[k]))

            self.bkc = bkc

            print('Calculation of the Matrix A...')
            A = SparseMatrix(Matrix([h]).jacobian(var))
            aval = []
            asubi = []
            asubj = []
            print('Reformat Matrix A')
            for k in range(A.RL.__len__()):
                asubi.append(A.RL[k][0])
                asubj.append(A.RL[k][1])
                aval.append(A.RL[k][2])
            aval = np.asarray(aval)
            self.asubi = asubi
            self.asubj = asubj
            print('A Matrix calculated.')
            print('Lambdify')

            aval_lam = lambdify([v, F_ini_param, ds, kappa, P_max, ax_max, ay_max, v_ini, v_end], aval,
                                modules=['numpy'])

            blc_lam = lambdify([v, F_ini_param, ds, kappa, P_max, ax_max, ay_max, v_ini, v_end], blc, modules=['numpy'])

            buc_lam = lambdify([v, F_ini_param, ds, kappa, P_max, ax_max, ay_max, v_ini, v_end], buc, modules=['numpy'])

            blx_lam = lambdify([v, F_ini_param, ds, kappa, P_max, ax_max, ay_max, v_ini, v_end], blx, modules=['numpy'])

            bux_lam = lambdify([v, F_ini_param, ds, kappa, P_max, ax_max, ay_max, v_ini, v_end], bux, modules=['numpy'])

            self.aval_lam = aval_lam
            self.blc_lam = blc_lam
            self.buc_lam = buc_lam
            self.blx_lam = blx_lam
            self.bux_lam = bux_lam

            # Save Lambdifiy Matrix and Vectors
            mod_local_trajectory_path = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            sys.path.append(mod_local_trajectory_path)
            if self.sol_options[self.key]['Friction'] == "Circle":
                file = "Mosek" + "_" + "point_mass" + "_" + str(N) + ".pkl"
            elif self.sol_options[self.key]['Friction'] == "Diamond":
                file = "Mosek" + "_" + "point_mass" + "_" + str(N) + "_" + str(
                    self.sol_options[self.key]['Friction']) + ".pkl"

            filepath = params_path + '/Lambdify_Function/'
            filename = filepath + file
            os.makedirs(filepath, exist_ok=True)

            dill.settings['recurse'] = True
            dill.dump([cval_lam, csubj, qval_lam, qsubi, qsubj, aval_lam, asubi,
                       asubj, blc_lam, buc_lam, bkc, blx_lam, bux_lam, bkx],
                      open(filename, "wb"))

    # Kinematic bicycle model
    def sol_init_km(self,
                    params_path: str):
        """Function to initialize the MOSEK solver with the kinematic-bicycle model \n
        by defining the objective function and constraints.\n
        Saves the matrix and vectors in order to re-use the same QP and avoid recalculations.

        :param params_path: absolute path to folder containing config file .ini

        :Authors:
            Tobias Klotz <tobias.klotz@tum.de>

        :Created on:
            16.06.2020
        """

        # Planing horizon
        N = self.m

        # Open Lambdify Matrix and Vectors if available
        if self.sol_options[self.key]['Friction'] == "Circle":
            file = "Mosek" + "_" + "kinematic_bicycle" + "_" + str(N) + ".pkl"
        elif self.sol_options[self.key]['Friction'] == "Diamond":
            file = "Mosek" + "_" + "kinematic_bicycle" + "_" + str(N) + "_" \
                   + str(self.sol_options[self.key]['Friction']) + ".pkl"

        filepath = params_path + '/Lambdify_Function/'
        filename = filepath + file

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
            self.cval_lam, self.csubj, self.qval_lam, self.qsubi, self.qsubj, self.aval_lam, self.asubi, self.asubj, \
                self.blc_lam, self.buc_lam, self.bkc, self.blx_lam, self.bux_lam, self.bkx \
                = dill.load(open(filename, "rb"))

        # Ceck if there is a Lambdification File
        else:
            inf = 0.0

            ############################################################################################################
            # Optimization Variables
            ############################################################################################################
            # velocity [m/s]
            v = symbols('v0:%d' % N)  # Velocity

            var = v

            ############################################################################################################
            # Online Parameters
            ############################################################################################################
            # curvature [1/m]
            kappa = symbols('kappa0:%d' % (N))
            # discretization step length [m]
            ds = symbols('ds:0%d' % (N - 1))
            # initial velocity [m/s]
            v_ini = symbols('v_ini')
            # end velocity [m/s]
            v_end = symbols('v_end')
            # initial force [kN]
            F_ini = symbols('F_ini')
            # max. power [kW]
            P_max = symbols('P_max0:%d' % (N))
            # max. acceleration in x-direction of the vehicle [m/s²]
            ax_max = symbols('ax_max0:%d' % (N))
            # max. acceleration in y-direction of the vehicle [m/s²]
            ay_max = symbols('ay_max0:%d' % (N))
            # Orientation [rad]
            psi = symbols('psi0:%d' % N)  # Orientation

            ############################################################################################################
            # Objective function
            ############################################################################################################

            # Cost Function: min s_i/v_i+1
            cost_func = 0

            for k in range(N - 1):
                cost_func += ds[k] / v[k + 1]

            # Calculate the Jacobi Function
            print('Calculation of the Jacobi Matrix')
            c = SparseMatrix(Matrix([cost_func]).jacobian(var))
            cval = []
            csubj = []
            for k in range(c.RL.__len__()):
                csubj.append(c.RL[k][1])
                cval.append(c.RL[k][2])

            cval_lam = lambdify([v, ds], cval, 'numpy')
            self.cval_lam = cval_lam
            self.csubj = csubj
            print('Calculation of the Hessian Matrix')

            q = SparseMatrix(hessian(cost_func, var))
            qval = []
            qsubi = []
            qsubj = []
            for k in range(q.RL.__len__()):
                qsubi.append(q.RL[k][0])
                qsubj.append(q.RL[k][1])
                qval.append(q.RL[k][2])

            qval_lam = lambdify([v, ds], qval, 'numpy')
            self.qval_lam = qval_lam
            self.qsubi = qsubi
            self.qsubj = qsubj

            ############################################################################################################
            # Constraints
            ############################################################################################################

            dt = []
            # Calculate Acceleration and Driving Force
            a = []
            F_dr = []
            delta = []
            for k in range(N - 1):
                a.append((v[k + 1] ** 2 - v[k] ** 2) / (2 * ds[k]))
                F_dr.append(self.Car.m * a[k] + 0.5 * self.Car.rho * self.Car.c_w * self.Car.A * v[k] ** 2)

            for k in range(N):
                delta.append(atan2(kappa[k] * self.Car.L, 1))

            for k in range(N - 1):
                dt.append(-v[k] / a[k] + ((v[k] / a[k]) ** 2 + 2 * ds[k] / a[k]) ** 0.5)

            # --- Boundary Conditions
            bkx = []
            blx = []
            bux = []
            # Velocity Constraint
            for k in range(len(v)):
                if k == 0:
                    bkx.append(mosek.boundkey.fx)
                    blx.append(0.0)
                    bux.append(0.0)
                elif k == N - 1:
                    bkx.append(mosek.boundkey.ra)
                    blx.append(0.0 - v[k])
                    bux.append(v_end - v[k])
                else:
                    bkx.append(mosek.boundkey.ra)
                    blx.append(0.0 - v[k])
                    bux.append(self.Car.v_max - v[k])

            self.bkx = bkx

            # --- CONSTRAINTS
            print('Calculation of the h Vector')
            h = []
            bkc = []
            blc = []
            buc = []

            # --- INEQUALITY CONSTRAINTS
            for k in range(N - 1):

                if self.sol_options[self.key]['Friction'] == "Circle":
                    # Kamm Circle
                    h = np.append(h, [a[k] ** 2 + (kappa[k] * v[k] ** 2) ** 2,
                                      ])
                    bkc.append(mosek.boundkey.up)
                    blc.append(- np.inf)
                    buc.append(self.Car.a_lat_max ** 2 - a[k] ** 2 - (kappa[k] * v[k] ** 2) ** 2)

                elif self.sol_options[self.key]['Friction'] == "Diamond":
                    # Friction Diamond
                    h = np.append(h, [a[k] / ax_max[k] + (kappa[k] * v[k] ** 2) / ay_max[k],
                                      a[k] / ax_max[k] - (kappa[k] * v[k] ** 2) / ay_max[k],
                                      - a[k] / ax_max[k] + (kappa[k] * v[k] ** 2) / ay_max[k],
                                      - a[k] / ax_max[k] - (kappa[k] * v[k] ** 2) / ay_max[k],
                                      ])
                    bkc.append(mosek.boundkey.up)
                    blc.append(- np.inf)
                    buc.append(1 - a[k] / ax_max[k] - (kappa[k] * v[k] ** 2) / ay_max[k])

                    bkc.append(mosek.boundkey.up)
                    blc.append(- np.inf)
                    buc.append(1 - a[k] / ax_max[k] + (kappa[k] * v[k] ** 2) / ay_max[k])

                    bkc.append(mosek.boundkey.up)
                    blc.append(- np.inf)
                    buc.append(1 + a[k] / ax_max[k] - (kappa[k] * v[k] ** 2) / ay_max[k])

                    bkc.append(mosek.boundkey.up)
                    blc.append(- np.inf)
                    buc.append(1 + a[k] / ax_max[k] + (kappa[k] * v[k] ** 2) / ay_max[k])

            # Power Constraint
            for k in range(N - 1):
                h = np.append(h, [F_dr[k] * v[k]])
                bkc.append(mosek.boundkey.up)
                blc.append(-inf)
                buc.append(P_max[k] - F_dr[k] * v[k])

            # Force Constraint
            for k in range(N - 1):
                h = np.append(h, [F_dr[k]])
                if k == 0:
                    bkc.append(mosek.boundkey.fx)
                    blc.append(F_ini - F_dr[k])
                    buc.append(F_ini - F_dr[k])

                else:
                    bkc.append(mosek.boundkey.ra)
                    blc.append(self.Car.F_br_max - F_dr[k])
                    buc.append(self.Car.F_dr_max - F_dr[k])

            # Force Derivative Constraint
            for k in range(N - 2):
                h = np.append(h, F_dr[k + 1] - F_dr[k])
                bkc.append(mosek.boundkey.ra)
                blc.append(- 4 - (F_dr[k + 1] - F_dr[k]))
                buc.append(4 - (F_dr[k + 1] - F_dr[k]))

            # Constraint of Steering Angle Derivative
            for k in range(N - 1):
                h = np.append(h, [(delta[k + 1] - delta[k]) * v[k] / ds[k]
                                  ])
                bkc.append(mosek.boundkey.ra)
                blc.append(- self.Car.delta_max - (delta[k + 1] - delta[k]) * v[k] / ds[k])
                buc.append(self.Car.delta_max - (delta[k + 1] - delta[k]) * v[k] / ds[k])

            self.bkc = bkc
            print('Calculation of the Matrix A...')
            A = SparseMatrix(Matrix([h]).jacobian(var))
            aval = []
            asubi = []
            asubj = []
            print('Reformat Matrix A')
            for k in range(A.RL.__len__()):
                asubi.append(A.RL[k][0])
                asubj.append(A.RL[k][1])
                aval.append(A.RL[k][2])
            aval = np.asarray(aval)
            self.asubi = asubi
            self.asubj = asubj
            print('A Matrix calculated.')

            # Lambdify Matrix and Vectors
            print('Lambdify')
            aval_lam = lambdify([v, ds, kappa, psi, F_ini, v_ini, v_end, P_max, ax_max, ay_max], aval,
                                modules=['numpy', 'math'])

            blc_lam = lambdify([v, ds, kappa, psi, F_ini, v_ini, v_end, P_max, ax_max, ay_max], blc,
                               modules=['numpy', 'math'])

            buc_lam = lambdify([v, ds, kappa, psi, F_ini, v_ini, v_end, P_max, ax_max, ay_max], buc,
                               modules=['numpy', 'math'])

            blx_lam = lambdify([v, F_ini, v_ini, v_end, P_max, ax_max, ay_max], blx, modules=['numpy', 'math'])

            bux_lam = lambdify([v, F_ini, v_ini, v_end, P_max, ax_max, ay_max], bux, modules=['numpy', 'math'])

            self.aval_lam = aval_lam
            self.blc_lam = blc_lam
            self.buc_lam = buc_lam
            self.blx_lam = blx_lam
            self.bux_lam = bux_lam

            # Save Lambdify Matrix and Vector
            mod_local_trajectory_path = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            sys.path.append(mod_local_trajectory_path)
            if self.sol_options[self.key]['Friction'] == "Circle":
                file = "Mosek" + "_" + "kinematic_bicycle" + "_" + str(N) + ".pkl"
            elif self.sol_options[self.key]['Friction'] == "Diamond":
                file = "Mosek" + "_" + "kinematic_bicycle" + "_" + str(N) + "_" + str(
                    self.sol_options[self.key]['Friction']) + ".pkl"
            filepath = params_path + '/Lambdify_Function/'
            filename = filepath + file

            dill.settings['recurse'] = True
            dill.dump([cval_lam, csubj, qval_lam, qsubi, qsubj, aval_lam, asubi,
                       asubj, blc_lam, buc_lam, bkc, blx_lam, bux_lam, bkx], open(filename, "wb"))

    # Dynamic bicycle model
    def sol_init_dm(self,
                    params_path: str):
        """Function to initialize the MOSEK solver with the dynamci-bicycle model \n
        by defining the objective function and constraints.\n
        Saves the matrix and vectors in order to re-use the same QP and avoid recalculations.

        :param params_path: absolute path to folder containing config file .ini

        :Authors:
            Tobias Klotz <tobias.klotz@tum.de>

        :Created on:
            16.06.2020
        """

        # Planing horizon
        N = self.m

        # Open Lambdify Matrix and Vector if available
        file = "Mosek" + "_" + "dynamic_bicycle" + "_" + str(N) + ".pkl"
        filepath = params_path + '/Lambdify_Function/'
        filename = filepath + file
        os.makedirs(filepath, exist_ok=True)

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
            self.cval_lam, self.csubj, self.qval_lam, self.qsubi, self.qsubj, self.aval_lam, self.asubi, self.asubj, \
                self.blc_lam, self.buc_lam, self.bkc, self.blx_lam, self.bux_lam, self.bkx \
                = dill.load(open(filename, "rb"))

        # Ceck if there is a Lambdification File
        else:
            ############################################################################################################
            # Optimization Variables
            ############################################################################################################
            # velocity [m/s]
            v = symbols('v0:%d' % N)  # Velocity
            # Slip Angle [rad]
            beta = symbols('beta0:%d' % N)
            # Gear Rate [rad/s]
            omega = symbols('omega0:%d' % N)
            # Driving Force [kN]
            F_dr = symbols('F_dr0:%d' % (N - 1))
            # Braking Force [kN]
            F_br = symbols('F_br0:%d' % (N - 1))
            # Steer Angle [rad]
            delta = symbols('delta0:%d' % (N - 1))
            # Optimization Vector
            var = v + beta + omega + F_dr + F_br + delta

            ############################################################################################################
            # Online Parameters
            ############################################################################################################
            # curvature [1/m]
            kappa = symbols('kappa0:%d' % (N))
            # discretization step length [m]
            ds = symbols('ds:0%d' % (N - 1))
            # initial velocity [m/s]
            v_ini = symbols('v_ini')
            # end velocity [m/s]
            v_end = symbols('v_end')
            # max. power [kW]
            P_max = symbols('P_max0:%d' % (N))
            # max. acceleration in x-direction of the vehicle [m/s²]
            ax_max = symbols('ax_max0:%d' % (N))
            # max. acceleration in y-direction of the vehicle [m/s²]
            ay_max = symbols('ay_max0:%d' % (N))

            ############################################################################################################
            # Objective function
            ############################################################################################################
            # Cost Function: min s_i/v_i+1
            cost_func = 0

            for k in range(N - 1):
                cost_func += ds[k] / v[k + 1]

            # Calculate the Jacobi Function
            print('Calculation of the Jacobi Matrix')
            c = SparseMatrix(Matrix([cost_func]).jacobian(var))
            cval = []
            csubj = []
            for k in range(c.RL.__len__()):
                csubj.append(c.RL[k][1])
                cval.append(c.RL[k][2])

            cval_lam = lambdify([v, ds], cval, 'numpy')
            self.cval_lam = cval_lam
            self.csubj = csubj
            print('Calculation of the Hessian Matrix')

            q = SparseMatrix(hessian(cost_func, var))
            qval = []
            qsubi = []
            qsubj = []
            for k in range(q.RL.__len__()):
                qsubi.append(q.RL[k][0])
                qsubj.append(q.RL[k][1])
                qval.append(q.RL[k][2])

            qval_lam = lambdify([v, ds], qval, 'numpy')
            self.qval_lam = qval_lam
            self.qsubi = qsubi
            self.qsubj = qsubj

            ############################################################################################################
            # Constraints
            ############################################################################################################
            # --- BOUNDARY CONDITONS for OPTIMIZATION vARIABLE
            bkx = []
            blx = []
            bux = []
            # Velocity Boundary Constraint
            for k in range(len(v)):
                if k == 0:
                    bkx.append(mosek.boundkey.ra)
                    blx.append(0.0)
                    bux.append(v_ini - v[k])
                elif k == len(v) - 1:
                    bkx.append(mosek.boundkey.ra)
                    blx.append(0.0 - v[k])
                    bux.append(v_end - v[k])
                else:
                    bkx.append(mosek.boundkey.ra)
                    blx.append(0.0 - v[k])
                    bux.append(self.Car.v_max - v[k])
            # Slip Angle Constraint
            for k in range(len(beta)):
                bkx.append(mosek.boundkey.ra)
                blx.append(- self.Car.beta_max - beta[k])
                bux.append(self.Car.beta_max - beta[k])
            # Yaw Rate Constraint
            for k in range(len(omega)):
                bkx.append(mosek.boundkey.ra)
                blx.append(- self.Car.omega_max - omega[k])
                bux.append(self.Car.omega_max - omega[k])
            # Driving Force Constraint
            for k in range(len(F_dr)):
                bkx.append(mosek.boundkey.ra)
                blx.append(0.0 - F_dr[k])
                bux.append(self.Car.F_dr_max - F_dr[k])
            # Braking Force Constraint
            for k in range(len(F_br)):
                bkx.append(mosek.boundkey.ra)
                blx.append(self.Car.F_br_max - F_br[k])
                bux.append(0.0 - F_br[k])
            # Steer Angle Constraint
            for k in range(len(delta)):
                bkx.append(mosek.boundkey.ra)
                blx.append(- self.Car.delta_max - delta[k])
                bux.append(self.Car.delta_max - delta[k])

            self.bkx = bkx

            # --- FORCES
            alpha_f = []
            alpha_r = []
            ma_x = []
            ma_y = []

            F_d = []
            # Rolling Friction Force [kN]
            F_roll_f = self.Car.f_r * self.Car.m * self.Car.g * self.Car.l_r / (self.Car.l_r + self.Car.l_f)
            F_roll_r = self.Car.f_r * self.Car.m * self.Car.g * self.Car.l_f / (self.Car.l_r + self.Car.l_f)

            F_xf = []
            F_xr = []
            F_yf = []
            F_yr = []
            F_zf = []
            F_zr = []

            for k in range(N - 1):
                # Tire Slip Angle (front/rear)
                alpha_f = np.append(alpha_f,
                                    delta[k] - atan2((self.Car.l_f * kappa[k] * v[k]
                                                      / (2 * np.pi) + v[k] * sin(beta[k])),
                                                     (v[k] * cos(beta[k]))))
                alpha_r = np.append(alpha_r, atan2((self.Car.l_r * kappa[k] * v[k] / (2 * np.pi) - v[k] * sin(beta[k])),
                                                   (v[k] * cos(beta[k]))))
                # aerodynamic resistance [kN]
                F_d = np.append(F_d, 0.5 * self.Car.c_w * self.Car.rho * self.Car.A * v[k] ** 2)

                # force at axle in x-direction (front & rear)
                F_xf = np.append(F_xf, self.Car.k_dr * F_dr[k] + self.Car.k_br * F_br[k] - F_roll_f)
                F_xr = np.append(F_xr, (1 - self.Car.k_dr) * F_dr[k] + (1 - self.Car.k_br) * F_br[k] - F_roll_r)

                # total force in x-direction at CoG
                ma_x = np.append(ma_x, F_xf[k] + F_xr[k] - F_d[k])

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
                                 / self.Car.F_z0 * sin(
                    self.Car.C_f * atan2(self.Car.B_f * alpha_f[k] - self.Car.E_f
                                         * (self.Car.B_f * alpha_f[k] - atan2(self.Car.B_f * alpha_f[k], 1)), 1)))
                F_yr = np.append(F_yr, self.Car.D_r * (1 + self.Car.eps_r * F_zr[k] / self.Car.F_z0) * F_zr[k]
                                 / self.Car.F_z0 * sin(
                    self.Car.C_r * atan2(self.Car.B_r * alpha_r[k] - self.Car.E_r
                                         * (self.Car.B_r * alpha_r[k] - atan2(self.Car.B_r * alpha_r[k], 1)), 1)))

                # total force in y-direction at CoG
                ma_y = np.append(ma_y, F_yr[k] + F_xf[k] * sin(delta[k]) + F_yf[k] * cos(delta[k]))

            bkc = []
            blc = []
            buc = []
            h = []
            # EQUALITY CONSTRAINTS
            for k in range(N - 1):
                h = np.append(h,
                              [  # Derivation of Velocity (Christ Eq. 5.2)
                                  v[k + 1] - v[k] - ds[k] / v[k]
                                  * (1 / self.Car.m * (+ F_xr[k] * cos(beta[k])
                                                       + F_xf[k] * cos(delta[k] - beta[k])
                                                       + F_yr[k] * sin(beta[k])
                                                       - F_yf[k] * sin(delta[k] - beta[k])
                                                       - F_d[k] * cos(beta[k]))),
                                  # Derivation of Slip Angle (Christ Eq. 5.3)
                                  (beta[k + 1] - beta[k]) / (ds[k] / v[k]) - 1 / (2 * np.pi)
                                  * (- kappa[k] * v[k] + 1 / (self.Car.m * v[k])
                                     * (- F_xr[k] * sin(beta[k])
                                        + F_xf[k] * sin(delta[k] - beta[k])
                                        + F_yr[k] * cos(beta[k])
                                        + F_yf[k] * cos(delta[k] - beta[k])
                                        + F_d[k] * sin(beta[k])))])
                # Derivation of Velocity (Christ Eq. 5.2)
                bkc.append(mosek.boundkey.fx)
                blc.append(v[k + 1] - v[k] - ds[k] / v[k]
                           * (1 / self.Car.m * (+ F_xr[k] * cos(beta[k])
                                                + F_xf[k] * cos(delta[k] - beta[k])
                                                + F_yr[k] * sin(beta[k])
                                                - F_yf[k] * sin(delta[k] - beta[k])
                                                - F_d[k] * cos(beta[k]))))
                buc.append(v[k + 1] - v[k] - ds[k] / v[k]
                           * (1 / self.Car.m * (+ F_xr[k] * cos(beta[k])
                                                + F_xf[k] * cos(delta[k] - beta[k])
                                                + F_yr[k] * sin(beta[k])
                                                - F_yf[k] * sin(delta[k] - beta[k])
                                                - F_d[k] * cos(beta[k]))))
                # Derivation of Slip Angle (Christ Eq. 5.3)
                bkc.append(mosek.boundkey.fx)
                blc.append((beta[k + 1] - beta[k]) / (ds[k] / v[k]) - 1 / (2 * np.pi)
                           * (- kappa[k] * v[k] + 1 / (self.Car.m * v[k])
                              * (- F_xr[k] * sin(beta[k])
                                 + F_xf[k] * sin(delta[k] - beta[k])
                                 + F_yr[k] * cos(beta[k])
                                 + F_yf[k] * cos(delta[k] - beta[k])
                                 + F_d[k] * sin(beta[k]))))
                buc.append((beta[k + 1] - beta[k]) / (ds[k] / v[k]) - 1 / (2 * np.pi)
                           * (- kappa[k] * v[k] + 1 / (self.Car.m * v[k])
                              * (- F_xr[k] * sin(beta[k])
                                 + F_xf[k] * sin(delta[k] - beta[k])
                                 + F_yr[k] * cos(beta[k])
                                 + F_yf[k] * cos(delta[k] - beta[k])
                                 + F_d[k] * sin(beta[k]))))

            # INEQUALITY CONSTRAINTS
            mu_x = []
            mu_y = []
            # Road Friction
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
                    # Power Constraint
                    bkc.append(mosek.boundkey.ra)
                    blc.append(- v[k] * F_dr[k])
                    buc.append(P_max[k] - v[k] * F_dr[k])
                    # Braking and Driving Force Constraint
                    bkc.append(mosek.boundkey.ra)
                    blc.append(- 0.02 - F_dr[k] * F_br[k])
                    buc.append(- F_dr[k] * F_br[k])
                    # Kamm Circle Front Axle
                    bkc.append(mosek.boundkey.ra)
                    blc.append(- (F_xf[k] / (mu_x[k] * F_zf[k])) ** 2 - (F_yf[k] / (mu_y[k] * F_zf[k])) ** 2)
                    buc.append(1.0 - (F_xf[k] / (mu_x[k] * F_zf[k])) ** 2 - (F_yf[k] / (mu_y[k] * F_zf[k])) ** 2)
                    # Kamm Circle Rear Axle
                    bkc.append(mosek.boundkey.ra)
                    blc.append(- (F_xr[k] / (mu_x[k] * F_zr[k])) ** 2 - (F_yr[k] / (mu_y[k] * F_zr[k])) ** 2)
                    buc.append(1.0 - (F_xr[k] / (mu_x[k] * F_zr[k])) ** 2 - (F_yr[k] / (mu_y[k] * F_zr[k])) ** 2)

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
                    # Power Constraint
                    bkc.append(mosek.boundkey.ra)
                    blc.append(- v[k] * F_dr[k])
                    buc.append(P_max[k] - v[k] * F_dr[k])
                    # Braking and Driving Force Constraint
                    bkc.append(mosek.boundkey.ra)
                    blc.append(- 0.02 - F_dr[k] * F_br[k])
                    buc.append(- F_dr[k] * F_br[k])
                    # Constraint Derivative of Driving Force
                    bkc.append(mosek.boundkey.up)
                    blc.append(- np.inf)
                    buc.append(self.Car.F_dr_max - ((F_dr[k + 1] - F_dr[k]) / (ds[k] / v[k])))
                    # Constraint Derivative of Braking Force
                    bkc.append(mosek.boundkey.lo)
                    blc.append(self.Car.F_br_max - ((F_br[k + 1] - F_br[k]) / (ds[k] / v[k])))
                    buc.append(np.inf)
                    # Constraint Derivative of Steer Angle
                    bkc.append(mosek.boundkey.ra)
                    blc.append(- self.Car.delta_max - ((delta[k + 1] - delta[k]) / (ds[k] / v[k])))
                    buc.append(self.Car.delta_max - ((delta[k + 1] - delta[k]) / (ds[k] / v[k])))
                    # Kamm Circle Front Axle
                    bkc.append(mosek.boundkey.ra)
                    blc.append(- (F_xf[k] / (mu_x[k] * F_zf[k])) ** 2 - (F_yf[k] / (mu_y[k] * F_zf[k])) ** 2)
                    buc.append(1.0 - (F_xf[k] / (mu_x[k] * F_zf[k])) ** 2 - (F_yf[k] / (mu_y[k] * F_zf[k])) ** 2)
                    # Kamm Circle Rear Axle
                    bkc.append(mosek.boundkey.ra)
                    blc.append(- (F_xr[k] / (mu_x[k] * F_zr[k])) ** 2 - (F_yr[k] / (mu_y[k] * F_zr[k])) ** 2)
                    buc.append(1.0 - (F_xr[k] / (mu_x[k] * F_zr[k])) ** 2 - (F_yr[k] / (mu_y[k] * F_zr[k])) ** 2)
                    '''
                    bkc.append(mosek.boundkey.ra)
                    blc.append(- self.Car.a_lat_max - 1 / self.Car.m * ma_y[k])
                    buc.append(self.Car.a_lat_max - 1 / self.Car.m * ma_y[k])'''

            self.bkc = bkc

            print('Calculation of the Matrix A...')
            A = SparseMatrix(Matrix([-h]).jacobian(var))
            aval = []
            asubi = []
            asubj = []
            print('Reformat Matrix A')
            for k in range(A.RL.__len__()):
                asubi.append(A.RL[k][0])
                asubj.append(A.RL[k][1])
                aval.append(A.RL[k][2])
            aval = np.asarray(aval)
            self.asubi = asubi
            self.asubj = asubj
            print('A Matrix calculated.')
            print('Lambdify')
            # Lambdify Matrix and Vector
            aval_lam = lambdify([v, beta, omega, F_dr, F_br, delta, ds, kappa, ax_max, ay_max, v_ini, P_max], aval,
                                modules=['numpy', 'math'])

            blc_lam = lambdify([v, beta, omega, F_dr, F_br, delta, ds, kappa, ax_max, ay_max, v_ini, P_max], blc,
                               modules=['numpy', 'math'])

            buc_lam = lambdify([v, beta, omega, F_dr, F_br, delta, ds, kappa, ax_max, ay_max, v_ini, P_max], buc,
                               modules=['numpy', 'math'])

            blx_lam = lambdify([v, beta, omega, F_dr, F_br, delta, ds, kappa, ax_max, ay_max, v_ini, P_max], blx,
                               modules=['numpy', 'math'])

            bux_lam = lambdify([v, beta, omega, F_dr, F_br, delta, ds, kappa, ax_max, ay_max, v_end, v_ini, P_max],
                               bux, modules=['numpy', 'math'])

            self.aval_lam = aval_lam
            self.blc_lam = blc_lam
            self.buc_lam = buc_lam
            self.blx_lam = blx_lam
            self.bux_lam = bux_lam

            # Save Lambdify Matrix and Vector
            mod_local_trajectory_path = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            sys.path.append(mod_local_trajectory_path)
            file = "Mosek" + "_" + "dynamic_bicycle" + "_" + str(N) + ".pkl"

            filepath = params_path + '/Lambdify_Function/'
            filename = filepath + file
            os.makedirs(filepath, exist_ok=True)

            dill.settings['recurse'] = True
            dill.dump([cval_lam, csubj, qval_lam, qsubi, qsubj, aval_lam, asubi,
                       asubj, blc_lam, buc_lam, bkc, blx_lam, bux_lam, bkx],
                      open(filename, "wb"))

    def calc_v_mosek(self,
                     N: int,
                     x0_v: np.ndarray,
                     F_ini: np.ndarray,
                     ds: np.ndarray,
                     kappa: np.ndarray,
                     P_max: float = None,
                     ax_max: np.array = None,
                     ay_max: np.array = None,
                     v_end: float = None,
                     v_max: np.array = None,
                     ):
        """Function to update the matrix and functions for the optimization problem. \n
        Solve the optimization problem with the solver OSQP.

        :param N: number of velocity points
        :param x0_v: initial guess velocity [m/s]
        :param F_ini: hard constrained initial force [kN]
        :param ds: discretization step length of given path [m]
        :param kappa: curvature profile of given path [rad/m]
        :param P_max: max. allowed power [kW]
        :param ax_max: max. allowed longitudinal acceleration [m/s^2]
        :param ay_max: max. allowed lateral accelereation [m/s]
        :param v_end: hard constrained max. allowed value of end velocity in optimization horizon [m/s]
        :param v_max: max. allowed velocity (in objective function) [m/s]

        :return: v: v: optimized velocity [m/s] \n
            F: optimize powertrain force [kN] \n
            P: optimized power force [kW] \n
            ax: acceleration in x-direction of CoG [m/s²] \n
            ay: acceleration in y-direction of CoG [m/s²]
            t_total: runtime of the solver OSQP [ms] \n
            sol_status: status of the solution (solved, infeasible, etc.)

        :Authors:
            Tobias Klotz <tobias.klotz@tum.de>

        :Created on:
            16.06.2020
        """

        # Point-mass model
        if self.sol_options[self.key]['Model'] == "PM":
            # Initialization of optimization variables
            if len(x0_v) == 0:
                v = np.ones(N)
            else:
                v = 0.9 * x0_v
                if x0_v[0] == 0:
                    v[0] = 0.1

            # initial velocity
            v_ini = 0.9 * x0_v[0]

            # Initialize max. power
            if self.sol_options[self.key]['VarPower']:
                pass
            else:
                P_max = self.Car.P_max * np.ones(N)

            # alpha = 1
            # Choose step length
            alpha = self.sol_options[self.key]['Alpha']

        # Kinematic bicycle model
        elif self.sol_options[self.key]['Model'] == "KM":
            # Calculate orientation from discretization step length and curvature
            psi = np.zeros(N)
            psi[0] = 0.0
            for k in range(len(kappa) - 1):
                psi[k + 1] = psi[k] + kappa[k] * ds[k]

            # initialize velocity
            if len(x0_v) == 0:
                v = np.ones(N)
            else:
                v = x0_v
                if x0_v[0] == 0:
                    v[0] = 0.1
            # initial velocity
            v_ini = x0_v[0]

            # initialize steering angle
            delta = []
            for k in range(N):
                delta.append(kappa[k] * self.Car.L)

            # Initialize max. power
            if self.sol_options[self.key]['VarPower']:
                pass
            else:
                P_max = self.Car.P_max * np.ones(N)

            # choose step length
            # alpha = 0.4
            # Choose step length
            alpha = self.sol_options[self.key]['Alpha']

        # Dynamci bicycle model
        elif self.sol_options[self.key]['Model'] == "DM":

            # initial guess of velocity
            if len(x0_v) == 0:
                v = np.ones(N)
            else:
                v = 0.9 * x0_v
                if x0_v[0] == 0:
                    v[0] = 0.1
            v = 5 * np.ones(N)
            v_ini = x0_v[0]
            # initial guess of slip angle
            beta = np.zeros(N)
            # initial guess of orientation
            omega = np.zeros(N)
            # initial guess of driving force
            F_dr = np.zeros(N - 1)
            # initial guess of braking force
            F_br = np.zeros(N - 1)
            # initial guess of steering angle
            delta = np.zeros(N - 1)
            # curvature
            kappa = kappa[0:N]
            # choose step length
            # Initialize max. power
            if self.sol_options[self.key]['VarPower']:
                pass
            else:
                P_max = self.Car.P_max * np.ones(N)
            # Initialize max. acceleartion
            if ax_max is None:
                ax_max = self.Car.a_max * np.ones(N)
            if ay_max is None:
                ay_max = self.Car.a_lat_max * np.ones(N)

            # alpha = 0.4
            # Choose step length
            alpha = self.sol_options[self.key]['Alpha']

        # estimate number of variables and constraints
        numvar = self.bkx.__len__()  # Number of Variables
        numcon = self.bkc.__len__()  # Number of Constraints

        var_list = np.arange(numvar)
        con_list = np.arange(numcon)

        # While loop for SQP sequences
        save_obj = []
        x_save = []
        t = []
        count = True
        infeasible = False
        no_solution = False
        optimal = False
        reset = False
        counter = 0
        while count:
            with mosek.Env() as env:
                with env.Task(0, 0) as task:

                    # Attach a printer to the task
                    task.set_Stream(mosek.streamtype.log, None)

                    task.appendvars(numvar)
                    task.appendcons(numcon)

                    # Fil objective
                    task.putqobj(self.qsubi, self.qsubj, self.qval_lam(v, ds))
                    c = self.cval_lam(v, ds)

                    for j in range(self.csubj.__len__()):
                        # Set the linear term c_j of the objective.
                        task.putcj(self.csubj[j], c[j])

                    # Fill Matrix and Vectors
                    if self.sol_options[self.key]['Model'] == "PM":
                        blx = self.blx_lam(v, F_ini, ds, kappa, P_max, ax_max, ay_max, v_ini, v_end)
                        bux = self.bux_lam(v, F_ini, ds, kappa, P_max, ax_max, ay_max, v_ini, v_end)
                        blc = self.blc_lam(v, F_ini, ds, kappa, P_max, ax_max, ay_max, v_ini, v_end)
                        buc = self.buc_lam(v, F_ini, ds, kappa, P_max, ax_max, ay_max, v_ini, v_end)
                        aval = self.aval_lam(v, F_ini, ds, kappa, P_max, ax_max, ay_max, v_ini, v_end)
                    elif self.sol_options[self.key]['Model'] == "KM":
                        blx = self.blx_lam(v, F_ini, v_ini, v_end, P_max, ax_max, ay_max)
                        bux = self.bux_lam(v, F_ini, v_ini, v_end, P_max, ax_max, ay_max)
                        blc = self.blc_lam(v, ds, kappa, psi, F_ini, v_ini, v_end, P_max, ax_max, ay_max)
                        buc = self.buc_lam(v, ds, kappa, psi, F_ini, v_ini, v_end, P_max, ax_max, ay_max)
                        aval = self.aval_lam(v, ds, kappa, psi, F_ini, v_ini, v_end, P_max, ax_max, ay_max)
                    elif self.sol_options[self.key]['Model'] == "DM":
                        blx = self.blx_lam(v, beta, omega, F_dr, F_br, delta, ds, kappa, ax_max, ay_max, v_ini, P_max)
                        bux = self.bux_lam(v, beta, omega, F_dr, F_br, delta, ds, kappa, ax_max, ay_max, v_end, v_ini,
                                           P_max)
                        blc = self.blc_lam(v, beta, omega, F_dr, F_br, delta, ds, kappa, ax_max, ay_max, v_ini, P_max)
                        buc = self.buc_lam(v, beta, omega, F_dr, F_br, delta, ds, kappa, ax_max, ay_max, v_ini, P_max)
                        aval = self.aval_lam(v, beta, omega, F_dr, F_br, delta, ds, kappa, ax_max, ay_max, v_ini, P_max)

                    task.putvarboundlist(var_list, self.bkx, blx, bux)

                    task.putconboundlist(con_list, self.bkc, blc, buc)

                    task.putaijlist(self.asubi, self.asubj, aval)

                    # Input the objective sense (minimize/maximize)
                    task.putobjsense(mosek.objsense.minimize)

                    # Optimize the task
                    t0 = time.perf_counter()
                    task.optimize()
                    t1 = time.perf_counter()

                    # save Calculation Time
                    t.append(t1 - t0)

                    # Output a solution
                    xx = [0.] * numvar
                    xc = [0.] * numcon
                    task.getxx(mosek.soltype.itr, xx)
                    task.getxc(mosek.soltype.itr, xc)

                    obj = task.getprimalobj(mosek.soltype.itr)
                    sol_status = task.getsolsta(mosek.soltype.itr)
                    print(sol_status)
                    # Adapt Variables:
                    x_array = np.asarray(xx)

                    # Count each iteration
                    counter += 1

                    # Stop after 30 iterations
                    if counter == 30:
                        t_total = sum(t)
                        count = False

                    # Check solution status and if optimal, adapt optimization variables
                    if self.sol_options[self.key]['Model'] == "PM":
                        if sol_status == mosek.solsta.optimal:
                            v += alpha * x_array[0:N]  # v(N): Velocity
                            optimal = True
                            no_solution = False
                            pass
                        else:
                            if optimal is True:
                                alpha_save = 0.5 * alpha
                                v += - alpha_save * x_save[0:N]  # v(N): Velocity

                                if no_solution is True:
                                    alpha = alpha * 0.5
                                no_solution = True
                            elif optimal is False:
                                # Reset:
                                if infeasible is False and reset is False:
                                    v = x0_v[0] * np.ones(N)
                                    reset = True
                                elif infeasible is False and reset is True:
                                    v = v_end * np.ones(N)
                                    v[0] = x0_v[0]

                        if save_obj:
                            if np.abs(obj) < 0.01:
                                if obj == 0:
                                    pass
                                else:
                                    t_total = sum(t)
                                    count = False

                        save_obj.append(obj)
                        if sol_status == mosek.solsta.optimal:
                            x_save = x_array

                    elif self.sol_options[self.key]['Model'] == "KM":
                        v += alpha * x_array[0:N]  # v(N): Velocity
                        if save_obj:
                            if np.abs(obj) < 0.01:
                                if obj == 0:
                                    pass
                                else:
                                    t_total = sum(t)
                                    count = False

                        save_obj.append(obj)
                        if sol_status == mosek.solsta.optimal or sol_status == mosek.solsta.unknown:
                            x_save = x_array

                    elif self.sol_options[self.key]['Model'] == "DM":

                        v += alpha * x_array[0:N]  # v(N): Velocity
                        beta += alpha * x_array[N:2 * N]  # beta(N): Slip Angle
                        omega += alpha * x_array[2 * N:3 * N]  # omega_z(N): Gear Rate
                        F_dr += alpha * x_array[3 * N:4 * N - 1]  # F_dr(N-1): Driving Force
                        F_br += alpha * x_array[4 * N - 1:5 * N - 2]  # F_br(N-1): Braking Force
                        delta += alpha * x_array[5 * N - 2:6 * N - 3]  # delta(N-1): Steering Angle

                        if save_obj:
                            if np.abs(obj) < 0.001:
                                if obj == 0:
                                    pass
                                else:
                                    t_total = sum(t)

                                    count = False
                            elif np.abs(obj) < 0.4:
                                alpha = 0.6

                        save_obj.append(obj)
                        if sol_status == mosek.solsta.optimal or sol_status == mosek.solsta.unknown:
                            x_save = x_array

        # Create Solution Vector
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

        v, F, P, ax, ay, F_xf, F_yf, F_xr, F_yr = self.transform_results(sol, ds, kappa, N)

        return v, F, P, ax, ay, F_xf, F_yf, F_xr, F_yr, t_total, sol_status

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
        # Initialize Force Vector
        F_xzf = []
        F_xzr = []
        F_yzf = []
        F_yzr = []

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
                                    delta[k] - atan2(
                                        (self.Car.l_f * kappa[k] * v[k] / (2 * np.pi) + v[k] * sin(beta[k])),
                                        (v[k] * cos(beta[k]))))
                alpha_r = np.append(alpha_r, atan2((self.Car.l_r * kappa[k] * v[k] / (2 * np.pi) - v[k] * sin(beta[k])),
                                                   (v[k] * cos(beta[k]))))

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
                F_yf = np.append(F_yf, self.Car.D_f * (1 + self.Car.eps_f * F_zf[k]
                                                       / self.Car.F_z0) * F_zf[k] / self.Car.F_z0
                                 * sin(self.Car.C_f * atan2(self.Car.B_f * alpha_f[k] - self.Car.E_f
                                                            * (self.Car.B_f * alpha_f[k]
                                                               - atan2(self.Car.B_f * alpha_f[k], 1)), 1)))
                F_yr = np.append(F_yr, self.Car.D_r * (1 + self.Car.eps_r * F_zr[k]
                                                       / self.Car.F_z0) * F_zr[k]
                                 / self.Car.F_z0
                                 * sin(self.Car.C_r * atan2(self.Car.B_r * alpha_r[k] - self.Car.E_r
                                                            * (self.Car.B_r * alpha_r[k]
                                                               - atan2(self.Car.B_r * alpha_r[k], 1)), 1)))

                # total force in y-direction at CoG
                ma_y = np.append(ma_y, F_yr[k] + F_xf[k] * sin(delta[k]) + F_yf[k] * cos(delta[k]))
                ay = ma_y / self.Car.m

                F = F_dr + F_br
                P = F * v[0:N - 1]

            # Front tire Constraint
            F_xzf = (F_xf / F_zf)
            F_yzf = (F_yf / F_zf)
            # Rear tire Constraint
            F_xzr = (F_xr / F_zr)
            F_yzr = (F_yr / F_zr)

        return v, F, P, ax, ay, np.array(F_xzf), np.array(F_yzf), np.array(F_xzr), np.array(F_yzr)


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
        self.a_max = opt_config.getfloat('VEHICLE', 'ay_max_mps2')  # [m/s²] Maximum Longitudinal Acceleration
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
