try:
    import casadi as cs
except ImportError:
    print('Warning: No module CasADi found. Not necessary on car but for development.')
import numpy as np
import time


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


if __name__ == '__main__':
    pass
