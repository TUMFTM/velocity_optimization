from src.VelQP import VelQP
from src.online_qp import online_qp
from src.params_vp_sqp import params_vp_sqp

"""
Python version: 3.5
Created by: Thomas Herrmann (thomas.herrmann@tum.de)
Created on: 01.11.2019

Documentation: Main script to test the velocity planner SQP.
"""

# --- Global settings for the main SQP test
m = 115
sid = 'PerfSQP'

# --- Create SymQP-instance
velqp = VelQP(m=m,
              sid=sid)

# --- Codegen
# symqp.sol_osqp.codegen('code', parameters='matrices', project_type='Unix Makefiles', force_rewrite=True)

# --- Get the initialized parameters
sqp_stgs,\
    v_ini,\
    v_max,\
    v_end,\
    x0_v,\
    x0_s_t,\
    F_ini,\
    kappa,\
    delta,\
    P_max,\
    ax_max,\
    ay_max,\
    err,\
    err_inf = params_vp_sqp(m=m, sid=sid)

v_op, s_t_op, qp_status = online_qp(velqp=velqp,
                                    v_ini=v_ini,
                                    kappa=kappa,
                                    delta_s=delta,
                                    P_max=P_max,
                                    ax_max=ax_max,
                                    ay_max=ay_max,
                                    x0_v=x0_v,
                                    v_max=v_max,
                                    v_end=v_end,
                                    F_ini=F_ini,
                                    s_glob=0)

if qp_status == 1:
    print("Optimal velocity: ", v_op)
    print("Slack variables on tires: ", s_t_op)

else:
    print("*** SQP found no valid solution for given input of ... ***")
    print("v_ini: ", v_ini)
    print("v_max: ", v_max)
    print("v_end: ", v_end)
    print("x0_v: ", x0_v)
    print("x0_s_t: ", x0_s_t)
    print("kappa: ", kappa)
    print("delta_s: ", delta)

    # --- Leave with error
    exit(1)
