import numpy as np
from velocity_optimization.src.VelQP import VelQP
from velocity_optimization.opt_postproc.src.VOptQPOASES import VOpt_qpOASES


def online_qp_postproc(velqp: VelQP,
                       vp_qpOASES: VOpt_qpOASES,
                       v_ini: float,
                       kappa: np.ndarray,
                       delta_s: np.ndarray,
                       P_max: np.array = None,
                       ax_max: np.array = None,
                       ay_max: np.array = None,
                       x0_v: np.ndarray = None,
                       x0_s_t: np.ndarray = None,
                       v_max: np.ndarray = None,
                       v_end: float = None,
                       F_ini: float = None,
                       s_glo: float = None) -> tuple:
    """
    Python version: 3.5
    Created by: Thomas Herrmann (thomas.herrmann@tum.de)
    Created on: 01.02.2020

    Documentation: Function to create an SQP that optimizes a velocity profile for a given path.
    For postprocessing and solver comparison purpose only!

    Inputs:
    velqp: Velocity optimization object using the OSQP solver
    vp_qpOASES: Velocity optimization object using the qpOASES solver
    v_ini: initial velocity [m/s]
    kappa: curvature profile [1/m]
    delta_s: spatial discretization between discretization points [m]
    P_max: max. allowed power [kW]
    ax_max: max. longitudinal acceleration limits [m/s^2]
    ay_max: max. lateral acceleration limits [m/s^2]
    x0_v: initial velocity guess [m/s]
    x0_s_t: initial slack value guess [m/s]
    v_max: max. velocity allowed [m/s]
    v_end: end velocity within optimization horizon [m/s]
    F_ini: initial force constraint from first to second velocity point to be optimized [kN]
    s_glo: global s coordinate on race track [m]

    Outputs:
    v_op: optimized velocity [m/s]
    s_t_op: optimized slack variables [-]
    F_op: optimized force profile [kN]
    """

    # --- Steplength reduction parameter for Armijo rule
    beta = 2 / 4

    ####################################################################################################################
    # --- Preparing input parameters for SQP
    ####################################################################################################################

    # --- Emergency SQP
    # upper bound for last velocity entry depending on current velocity
    if velqp.sid == 'EmergSQP':
        # --- Upper bound for last velocity point
        v_end = 0.4

        v_max = np.array(
            [v_ini - (x + 1) * 4 for x in range(velqp.sqp_stgs['m'])])
        # set all values below threshold to threshold
        v_max[v_max < 0.0] = 0.0

        # --- Assume linear velocity decrease to v_end from current velocity
        x0_v = np.array(
            [v_ini + x * (v_max[- 1] - v_ini) / velqp.sqp_stgs['m'] for x in range(velqp.sqp_stgs['m'])])

        # Initialize slack variables with zero values
        x0_s_t = np.array([0] * velqp.n)

        # Overwrite None
        F_ini = 0

    # --- /Emergency SQP

    # --- Performance SQP
    else:
        x0_s_t = np.array([0] * velqp.n)
    # --- /Performance SQP

    # --- Make sure to always have a numeric velocity > 0.1
    x0_v[0] = v_ini  # make sure, x0_v[0] == v_ini

    # --- Initialization
    #  optimization variables
    v_op = np.zeros(velqp.sqp_stgs['m'], )
    s_t_op = np.zeros(velqp.n, )

    # SQP mean-error
    err = np.inf
    # SQP infinity-error
    err_inf = np.inf

    # SQP-counter
    n_sqp = 0

    # --- Start SQP-loop
    while err > velqp.err or err_inf > velqp.err_inf:

        _, \
            q, \
            Am, \
            lo, \
            up = velqp.get_osqp_mat(x0_v=x0_v,
                                    x0_s_t=x0_s_t,
                                    v_ini=v_ini,
                                    v_max=v_max,
                                    v_end=v_end,
                                    F_ini=F_ini,
                                    P_max=P_max,
                                    kappa=kappa,
                                    delta_s=delta_s,
                                    ax_max=ax_max,
                                    ay_max=ay_max)

        # --- Initial guess for qpOASES
        x0_qpoases = list()
        x0_qpoases.append(x0_v)
        x0_qpoases.append(x0_s_t)
        # --- Conversion of format
        x0_qpoases = np.concatenate(x0_qpoases)

        # --- Solve the QP
        sol = vp_qpOASES.solve(x0=x0_qpoases[1:],
                               Hm=velqp.J_Hess[1:, 1:],
                               gv=q,
                               Am=Am,  # velqp.Am is a dense matrix --> take return value of get_osqp_mat
                               lba=lo,
                               uba=up)

        try:
            sol = np.array(sol)
        except ValueError:
            sol = np.ones((len(x0_v) + len(x0_s_t) - 1,))

        # --- Store solution from previous SQP-iteration
        o_old = np.append(v_op, x0_s_t)
        # --- Store errors from previous SQP-iteration
        err_old = err
        err_inf_old = err_inf

        ############################################################################################################
        # --- Armijo: decrease steplength alpha if SQP-error increases between iterations
        ############################################################################################################
        k = 0  # counter for Armijo-loop
        while True:
            # --- Choose a steplength
            alpha = beta ** k  # steplength e {1, beta, beta^2, ...}

            # --- Restructure solution from QP-solution
            # Add leading "0" as ini-velocity must be kept constant as given
            v_op = alpha * np.insert(sol[0:velqp.m - 1], 0, 0) + x0_v

            s_t_op = alpha * sol[velqp.m - 1:].reshape(sol[velqp.m - 1:].shape[0],) + x0_s_t

            # --- Calculate SQP iteration error
            o = np.append(v_op, s_t_op)
            err = np.sqrt(np.matmul(o - o_old, o - o_old)) / o.shape[0]
            err_inf = np.max(np.abs(o - o_old))

            # --- Break Armijo-loop in case a suitable steplength alpha was found
            if err < err_old and err_inf < err_inf_old:
                break
            # --- Increase Armijo-loop's counter and restart loop
            else:
                k += 1

        ############################################################################################################
        # --- Postprocessing Armijo-loop: Create new operating point for optimization variables
        ############################################################################################################
        # --- Create new operating-point for velocity variables
        x0_v = v_op
        # --- Create new operating-point for tire slack variables
        x0_s_t = s_t_op

        ################################################################################################################
        # --- Check termination criteria for SQP-loop
        ################################################################################################################
        # increase SQP-iteration counter
        n_sqp += 1

    ####################################################################################################################
    # --- Postprocessing with optimal solution form SQP
    ####################################################################################################################
    # Acceleration [m/s^2]
    acc = (v_op[1:] ** 2 - v_op[: - 1] ** 2) / (2 * delta_s)

    # --- Powertrain force [kN]
    F_op = velqp.sym_sc_['m_t_'] * acc + \
        velqp.sym_sc_['c_res_'] * (v_op ** 2)[0: velqp.m - 1] * 0.001

    print(velqp.sid + ' | nSQP (qpOASES): ' + str(n_sqp))

    return v_op, s_t_op, F_op
