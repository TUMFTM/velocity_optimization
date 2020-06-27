import numpy as np
import copy
import time
import datetime
import json

from velocity_optimization.src.VelQP import VelQP


def online_qp(velqp: VelQP,
              v_ini: float,
              kappa: np.ndarray,
              delta_s: np.ndarray,
              P_max: np.array = None,
              ax_max: np.array = None,
              ay_max: np.array = None,
              x0_v: np.ndarray = None,
              v_max: np.ndarray = None,
              v_end: float = None,
              F_ini: float = None,
              s_glob: float = None,
              v_max_cstr: np.ndarray = None) -> tuple:

    """
    Python version: 3.5
    Created by: Thomas Herrmann (thomas.herrmann@tum.de)
    Created on: 01.11.2019

    Documentation: Creates an SQP that optimizes a velocity profile for a given path.

    Inputs:
    velqp: QP solver object used within the SQP
    v_ini: initial velocity hard constraint [m/s]
    kappa: kappa profile of given path [rad/m]
    delta_s: discretization step length [m]
    P_max: max. allowed power [kW]
    ax_max: max. allowed longitudinal acceleration [m/s^2]
    ay_max: max. allowed ongitudial acceleration [m/s^2]
    x0_v: initial guess of optimal velocity [m/s]
    v_max: max. should velocity (objective function) [m/s]
    v_end: constrained end velocity in optimization horizon [m/s]
    F_ini: initial force constraint [kN]
    s_glob: global s coordinate of current vehicle position [m]
    v_max_cstr: max. must velocity (hard constraint) [m/s]

    Outputs:
    v_op: optimized velocity using OSQP as QP solver [m/s]
    s_t_op: optimized slack values [-]
    qp_status: status of last QP within SQP [-]
    """

    # --- Steplength reduction parameter for Armijo rule
    beta = 2 / 4

    # --- Initialization of logging variables
    x0_v_log = None
    x0_s_t_log = None
    kappa_log = None
    delta_s_log = None
    ax_max_log = None
    ay_max_log = None
    v_ini_log = None
    v_max_log = None
    v_end_log = None
    F_ini_log = None

    ####################################################################################################################
    # --- Preparing input parameters for SQP
    ####################################################################################################################

    # --- Emergency SQP
    # upper bound for last velocity entry depending on current velocity
    if velqp.sid == 'EmergSQP':
        # --- Upper bound for last velocity point
        v_end = 0.4

        v_max = np.array([v_ini - (x + 1) * 4 for x in range(velqp.sqp_stgs['m'])])
        # set all values below threshold to threshold
        v_max[v_max < 0.0] = 0.0

        # --- Assume linear velocity decrease to v_end from current velocity
        x0_v = np.array(
            [v_ini + x * (v_max[-1] - v_ini) / velqp.sqp_stgs['m'] for x in range(velqp.sqp_stgs['m'])])

        # Initialize slack variables with zero values
        x0_s_t = np.array([0] * velqp.n)

        # Overwrite None
        F_ini = 0
        s_glob = 0

    # --- /Emergency SQP

    # --- Performance SQP
    else:
        x0_s_t = np.array([0] * velqp.n)
    # --- /Performance SQP

    # --- Make sure to always have a numeric velocity > 0.1
    if v_ini < velqp.sym_sc_['vmin_mps_']:
        v_ini = velqp.sym_sc_['vmin_mps_']
    x0_v[0] = v_ini

    # --- Initialization of optimization variables
    # velocity [m/s]
    v_op = np.zeros(velqp.sqp_stgs['m'], )
    # slack variable on tires
    s_t_op = np.zeros(velqp.n, )

    qp_iter = 0
    qp_status = 0

    # SQP mean-error
    err = np.inf
    # SQP infinity-error
    err_inf = np.inf

    # SQP-counter
    n_sqp = 0
    # SQP-timer
    dt = 0

    # time limit of SQP loop [s]
    dt_lim = velqp.sqp_stgs['t_sqp_max']

    # --- Save inputs to SQP here for logging purpose
    x0_v_log = copy.deepcopy(x0_v.tolist())
    x0_s_t_log = copy.deepcopy(x0_s_t.tolist())
    kappa_log = kappa.tolist()
    delta_s_log = delta_s.tolist()
    v_ini_log = v_ini
    try:
        v_max_log = copy.deepcopy(v_max.tolist())
    except AttributeError:
        v_max_log = copy.deepcopy(v_max)

    v_end_log = v_end
    F_ini_log = F_ini

    if isinstance(ax_max, np.ndarray) and isinstance(ay_max, np.ndarray):
        ax_max_log = ax_max.tolist()
        ay_max_log = ay_max.tolist()
    else:
        ax_max_log = velqp.sym_sc_['axmax_mps2_']
        ay_max_log = velqp.sym_sc_['aymax_mps2_']

    if isinstance(P_max, np.ndarray):
        Pmax_log = P_max.tolist()
    else:
        Pmax_log = velqp.sym_sc_['Pmax_kW_']

    # --- Start SQP-loop
    t_start = time.time()
    while (err > velqp.err or err_inf > velqp.err_inf) and n_sqp < velqp.sqp_stgs['n_sqp_max'] and dt < dt_lim:

        # --- Update parameters of QP
        if len(x0_v) is not int(velqp.sqp_stgs['m']):
            print("Error in x0-length in ", velqp.sid)
            print(x0_v)
        if len(v_max) is not int(velqp.sqp_stgs['m']):
            print("Error in v_max-length in ", velqp.sid)
            print(v_max)

        # --- Update QP matrices
        velqp.osqp_update_online(x0_v=x0_v,
                                 x0_s_t=x0_s_t,
                                 v_ini=v_ini,
                                 v_max=v_max,
                                 v_end=v_end,
                                 F_ini=F_ini,
                                 kappa=kappa,
                                 delta_s=delta_s,
                                 P_max=P_max,
                                 ax_max=ax_max,
                                 ay_max=ay_max,
                                 vmax_cstr=v_max_cstr)

        # --- Solve the QP
        sol, qp_iter, qp_status = velqp.osqp_solve()

        # --- Check primal infeasibility of problem and return v = 0
        if qp_status == -3:
            break

        # --- Store solution from previous SQP-iteration
        o_old = np.append(v_op, x0_s_t)
        # --- Store errors from previous SQP-iteration
        err_old = err
        err_inf_old = err_inf
        try:
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

                s_t_op = alpha * sol[velqp.m - 1:] + x0_s_t

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

                if velqp.sqp_stgs['b_print_sqp_alpha']:
                    print(velqp.sid + " | alpha: " + str(alpha))

            ############################################################################################################
            # --- Postprocessing Armijo-loop: Create new operating point for optimization variables
            ############################################################################################################
            # --- Create new operating-point for velocity variables
            x0_v = v_op
            # --- Create new operating-point for tire slack variables
            x0_s_t = s_t_op

        except TypeError:
            # --- Do different initialization till n_sqp_max is reached
            if velqp.sid == 'EmergSQP':
                x0_v = (v_ini - 0.05) * np.ones((velqp.m, ))
                v_op = np.zeros(velqp.sqp_stgs['m'], )
                x0_s_t = np.zeros(velqp.n, )
                s_t_op = np.zeros(velqp.n, )
                print("No solution for emerg. line found. Retrying with different initialization ...")

                # Reset SQP-counter
                n_sqp = 0

        if not velqp.sqp_stgs['b_online_mode']:
            print('Optimized velocity profile: ', v_op[0:velqp.sqp_stgs['m']])
            if velqp.sqp_stgs['obj_func'] == 'slacks':
                print('Slacks on velocity: ', v_op[velqp.sqp_stgs['m']:2 * velqp.sqp_stgs['m']])

        if velqp.sqp_stgs['b_print_sqp_err']:
            print(velqp.sid + " | SQP err: " + str(err))
            print(velqp.sid + " | SQP inf.-err: " + str(err_inf))

        ################################################################################################################
        # --- Check termination criteria for SQP-loop
        ################################################################################################################
        # increase SQP-iteration counter
        n_sqp += 1

        if n_sqp >= velqp.sqp_stgs['n_sqp_max']:
            print(velqp.sid + " reached max. SQP iteration-number!")

        # update timer
        dt = time.time() - t_start
        if dt >= dt_lim:
            print(velqp.sid + " took too long!")

    if velqp.sqp_stgs['b_print_SQP_runtime']:
        print(velqp.sid + " | SQP time [ms]: " + str(dt * 1000))

    # Only write to log-file after SQP-iterations
    if velqp.sid == 'PerfSQP' and velqp.logger_perf is not None:
        velqp.logger_perf.debug('%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s',
                                str(datetime.datetime.now().time()),
                                s_glob,
                                json.dumps(x0_v_log),
                                json.dumps(x0_s_t_log),
                                json.dumps(kappa_log),
                                json.dumps(delta_s_log),
                                v_ini_log,
                                json.dumps(v_max_log),
                                v_end_log,
                                F_ini_log,
                                Pmax_log,
                                json.dumps(qp_iter),
                                json.dumps(qp_status),
                                json.dumps(dt * 1000))

        velqp.logger_perf.debug('%s', v_op.tolist())

        velqp.logger_perf.debug('%s', s_t_op.tolist())

        velqp.logger_perf.debug('%s;%s', ax_max_log, ay_max_log)

    elif velqp.sid == 'EmergSQP' and velqp.logger_emerg is not None:
        velqp.logger_emerg.debug('%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s',
                                 str(datetime.datetime.now().time()),
                                 s_glob,
                                 json.dumps(x0_v_log),
                                 json.dumps(x0_s_t_log),
                                 json.dumps(kappa_log),
                                 json.dumps(delta_s_log),
                                 v_ini_log,
                                 json.dumps(v_max_log),
                                 v_end_log,
                                 F_ini_log,
                                 Pmax_log,
                                 json.dumps(qp_iter),
                                 json.dumps(qp_status),
                                 json.dumps(dt * 1000))

        velqp.logger_emerg.debug('%s', v_op.tolist())

        velqp.logger_emerg.debug('%s', s_t_op.tolist())

        velqp.logger_emerg.debug('%s;%s', ax_max_log, ay_max_log)

    if velqp.sqp_stgs['b_trajectory_check']:
        ax_norm = np.abs(kappa[0:velqp.sqp_stgs['m'] - 1] * v_op[0:velqp.sqp_stgs['m'] - 1] ** 2) / 13.5
        ay_norm = np.abs(
            (v_op[1:velqp.sqp_stgs['m']] ** 2 - v_op[0:velqp.sqp_stgs['m'] - 1] ** 2) / (2 * np.array(delta_s))
            + (velqp.sym_sc_['c_res_'] * v_op[0:velqp.sqp_stgs['m'] - 1] ** 2) / (1000 * velqp.sym_sc_['m_t_'])) / 13.5

        perf_check = (ax_norm + ay_norm) > 1

        if perf_check[:-1].any():
            print(ax_norm + ay_norm)
            print('*** SQP: Trajectory not OK! ', velqp.sid, ' ***')

    if velqp.sqp_stgs['b_print_n_sqp']:
        print(velqp.sid + ' | nSQP ' + str(n_sqp))

    if velqp.sqp_stgs['b_print_s_v_val']:
        print(velqp.sid + ' | s_v_tires ' + str(x0_s_t))

    if velqp.sqp_stgs['b_print_J']:
        print("(v - v_max) ** 2", np.sum((v_op - v_max) ** 2))
        print("Tre. slacks", velqp.sym_sc_['s_tre_w_'] * np.sum(s_t_op ** 2))

    return v_op, s_t_op, qp_status
