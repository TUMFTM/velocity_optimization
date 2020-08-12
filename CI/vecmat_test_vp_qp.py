import numpy as np
import os
import velocity_optimization as vo

########################################################################################################################
# --- Author: Thomas Herrmann (thomas.herrmann@tum.de)
# --- Docu: This function checks the manual vector-matrix notation of the velocity-planner SQP against the
# numeric values that are calculated by substituting the symbolically deviated expressions
########################################################################################################################
# TODO: unit-check of filled matrices

# --- Conversion and test function
def conv_test(d1, d2):
    perc = 1e-4
    upper = lambda x: np.abs(x * perc) + x  # noqa: E731
    lower = lambda x: x - np.abs(x * perc)  # noqa: E731
    ufunc = np.vectorize(upper)
    lfunc = np.vectorize(lower)
    compI = ufunc(d1).astype(np.float32) >= d2.astype(np.float32)
    compII = lfunc(d1).astype(np.float32) <= d2.astype(np.float32)
    comp = compI & compII
    return comp


# --- Global parameters for Vector-Matrix Check
# m, random integer from 15 .. 115
m = 15 + np.random.randint(100)
# m = 11  # fix value for manual testing
sid = 'PerfSQP'

params_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'params/')
input_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'inputs/veh_dyn_info/')
logging_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs/vp_sqp')

# --- Create SymQP-instance
symqp = vo.src.SymQP.SymQP(m=m,
                           sid=sid,
                           params_path=params_path)

# --- Create VelQP-instance
velqp = vo.src.VelQP.VelQP(m=m,
                           sid=sid,
                           params_path=params_path,
                           input_path=input_path,
                           logging_path=logging_path,
                           ci=True)

# TODO: vary hard coded parameters here!
# --- Variation of parameters
# v_ini, random from 0 to 20
vini_mps_ = np.random.randint(20)

# v_end, random from vini + (0 to 10)
vend_mps_ = vini_mps_ + np.random.randint(10)

# x0_v, 30 * random from 0 .. 1
x0_ = 30 * np.random.rand(m, )
x0_[0] = vini_mps_

x0_s_t_ = np.random.rand(symqp.n, )

# F_ini, random from 0 .. 1
Fini_kN_ = np.random.rand(1, 1)

# Fini_tol in range from 0 .. 1
Fini_tol = np.random.rand(1, 1)

# v_max
vmax_mps_ = 50 + 20 * np.random.rand(m, )

# kappa
kappa_ = 0.001 * np.random.rand(m, ) - 0.002 * np.random.rand(m, )

# delta_s
delta_s_ = 2 + 0.5 * np.random.rand(m - 1, )

# var power limits from 270 * 0 .. 1
if symqp.sqp_stgs['b_var_power']:
    Pmax_ = 270 * np.random.rand(m - 1, )
else:
    Pmax_ = None

# var friction limits from 12.5 * 0 .. 1
if symqp.sqp_stgs['b_var_friction']:
    axmax_ = 12.5 * np.random.rand(m, )
    aymax_ = 12.5 * np.random.rand(m, )
else:
    axmax_ = None
    aymax_ = None

# --- Update solver values with generated random values
velqp.get_osqp_mat(x0_v=x0_,
                   x0_s_t=x0_s_t_,
                   v_ini=vini_mps_,
                   v_max=vmax_mps_,
                   v_end=vend_mps_,
                   F_ini=Fini_kN_,
                   kappa=kappa_,
                   delta_s=delta_s_,
                   P_max=Pmax_,
                   ax_max=axmax_,
                   ay_max=aymax_)

# --- Compare lambda-functions of expressions with raw vec.-mat.-notation
t = conv_test(symqp.fJ_Hess(x0_, symqp.sym_sc_['dvdv_w_'], symqp.sym_sc_['s_tre_w_lin_'],
              symqp.sym_sc_['s_tre_w_quad_'], symqp.sym_sc_['s_v_t_unit_']),
              velqp.J_Hess)
if not t.all():
    print('Error in J_hess!')
    exit(1)

t = conv_test(
    symqp.fJ_jac(
        x0_, x0_s_t_, vmax_mps_, symqp.sym_sc_['dvdv_w_'], symqp.sym_sc_['s_tre_w_lin_'],
        symqp.sym_sc_['s_tre_w_quad_'], symqp.sym_sc_['s_v_t_unit_']),
    velqp.J_jac)
if not t.all():
    print('Error in J_jac!')
    exit(1)

t = conv_test(symqp.fF_cst(x0_, delta_s_), velqp.F_cst)
if not t.all():
    print('Error in F_cst!')
    exit(1)

t = conv_test(symqp.fF_cst_jac(x0_, delta_s_), velqp.F_cst_jac)
if not t.all():
    print('Error in F_cst_jac!')
    exit(1)

t = conv_test(symqp.fF_ini_cst(x0_, delta_s_, Fini_kN_, Fini_tol), velqp.F_ini_cst)
if not t.all():
    print('Error in F_ini_cst!')
    exit(1)

t = conv_test(symqp.fF_ini_cst_jac(x0_, delta_s_), velqp.F_ini_cst_jac)
if not t.all():
    print('Error in F_ini_cst_jac!')
    exit(1)

t = conv_test(symqp.fv_cst_end(x0_, vend_mps_), velqp.v_cst_end)
if not t.all():
    print('Error in v_cst_end!')
    exit(1)

t = conv_test(symqp.fv_cst_end_jac(), velqp.v_cst_end_jac)
if not t.all():
    print('Error in v_cst_end_jac!')
    exit(1)

if symqp.sqp_stgs['b_var_power']:
    t = conv_test(symqp.fP_cst(x0_, delta_s_, Pmax_), velqp.P_cst)
else:
    t = conv_test(symqp.fP_cst(x0_, delta_s_), velqp.P_cst)
if not t.all():
    print('Error in P_cst!')
    exit(1)

t = conv_test(symqp.fP_cst_jac(x0_, delta_s_), velqp.P_cst_jac)
if not t.all():
    print('Error in P_cst_jac!')
    exit(1)

if symqp.sqp_stgs['b_var_friction']:
    t = conv_test(symqp.fTre_cst1(x0_, x0_s_t_, kappa_, delta_s_, axmax_, aymax_), velqp.Tre_cst1)
    if not t.all():
        print('Error in Tre_cst_1!')
        exit(1)

    t = conv_test(symqp.fTre_cst2(x0_, x0_s_t_, kappa_, delta_s_, axmax_, aymax_), velqp.Tre_cst2)
    if not t.all():
        print('Error in Tre_cst_2!')
        exit(1)

    t = conv_test(symqp.fTre_cst3(x0_, x0_s_t_, kappa_, delta_s_, axmax_, aymax_), velqp.Tre_cst3)
    if not t.all():
        print('Error in Tre_cst_3!')
        exit(1)

    t = conv_test(symqp.fTre_cst4(x0_, x0_s_t_, kappa_, delta_s_, axmax_, aymax_), velqp.Tre_cst4)
    if not t.all():
        print('Error in Tre_cst_4!')
        exit(1)

    t = conv_test(symqp.fTre_cst1_jac(x0_, x0_s_t_, kappa_, delta_s_, axmax_, aymax_), velqp.Tre_cst1_jac)
    if not t.all():
        print('Error in Tre_cst_1_jac!')
        exit(1)

    t = conv_test(symqp.fTre_cst2_jac(x0_, x0_s_t_, kappa_, delta_s_, axmax_, aymax_), velqp.Tre_cst2_jac)
    if not t.all():
        print('Error in Tre_cst_2_jac!')
        exit(1)

    t = conv_test(symqp.fTre_cst3_jac(x0_, x0_s_t_, kappa_, delta_s_, axmax_, aymax_), velqp.Tre_cst3_jac)
    if not t.all():
        print('Error in Tre_cst_3_jac!')
        exit(1)

    t = conv_test(symqp.fTre_cst4_jac(x0_, x0_s_t_, kappa_, delta_s_, axmax_, aymax_), velqp.Tre_cst4_jac)
    if not t.all():
        print('Error in Tre_cst_4_jac!')
        exit(1)

else:
    t = conv_test(symqp.fTre_cst1(x0_, x0_s_t_, kappa_, delta_s_), velqp.Tre_cst1)
    if not t.all():
        print('Error in Tre_cst_1!')
        exit(1)

    t = conv_test(symqp.fTre_cst2(x0_, x0_s_t_, kappa_, delta_s_), velqp.Tre_cst2)
    if not t.all():
        print('Error in Tre_cst_2!')
        exit(1)

    t = conv_test(symqp.fTre_cst3(x0_, x0_s_t_, kappa_, delta_s_), velqp.Tre_cst3)
    if not t.all():
        print('Error in Tre_cst_3!')
        exit(1)

    t = conv_test(symqp.fTre_cst4(x0_, x0_s_t_, kappa_, delta_s_), velqp.Tre_cst4)
    if not t.all():
        print('Error in Tre_cst_4!')
        exit(1)

    t = conv_test(symqp.fTre_cst1_jac(x0_, x0_s_t_, kappa_, delta_s_), velqp.Tre_cst1_jac)
    if not t.all():
        print('Error in Tre_cst_1_jac!')
        exit(1)

    t = conv_test(symqp.fTre_cst2_jac(x0_, x0_s_t_, kappa_, delta_s_), velqp.Tre_cst2_jac)
    if not t.all():
        print('Error in Tre_cst_2_jac!')
        exit(1)

    t = conv_test(symqp.fTre_cst3_jac(x0_, x0_s_t_, kappa_, delta_s_), velqp.Tre_cst3_jac)
    if not t.all():
        print('Error in Tre_cst_3_jac!')
        exit(1)

    t = conv_test(symqp.fTre_cst4_jac(x0_, x0_s_t_, kappa_, delta_s_), velqp.Tre_cst4_jac)
    if not t.all():
        print('Error in Tre_cst_4_jac!')
        exit(1)

t = conv_test(symqp.fdF_cst(x0_, delta_s_), velqp.dF_cst)
if not t.all():
    print('Error in dF_cst!')
    exit(1)

t = conv_test(symqp.fdF_cst_jac(x0_, delta_s_), velqp.dF_cst_jac)
if not t.all():
    print('Error in dF_cst_jac!')
    exit(1)

########################################################################################################################
# --- Check sparse CSC matrix fill against dense version of constraint matrix A
########################################################################################################################

# --- Deactivate sparse matrix fill to update the DENSE version
velqp.sqp_stgs['b_sparse_matrix_fill'] = 0

# --- Update solver values with generated random values ** DENSE VERSION **
velqp.get_osqp_mat(x0_v=x0_,
                   x0_s_t=x0_s_t_,
                   v_ini=vini_mps_,
                   v_max=vmax_mps_,
                   v_end=vend_mps_,
                   F_ini=Fini_kN_,
                   kappa=kappa_,
                   delta_s=delta_s_,
                   P_max=Pmax_,
                   ax_max=axmax_,
                   ay_max=aymax_)

# --- Activate sparse matrix fill to update the SPARSE version
velqp.sqp_stgs['b_sparse_matrix_fill'] = 1

# --- Update solver values with generated random values ** SPARSE VERSION **
velqp.get_osqp_mat(x0_v=x0_,
                   x0_s_t=x0_s_t_,
                   v_ini=vini_mps_,
                   v_max=vmax_mps_,
                   v_end=vend_mps_,
                   F_ini=Fini_kN_,
                   kappa=kappa_,
                   delta_s=delta_s_,
                   P_max=Pmax_,
                   ax_max=axmax_,
                   ay_max=aymax_)

if not (velqp.Am == velqp.Am_csc).all():
    print('Error in sparse CSC matrix!')
    exit(1)

########################################################################################################################
# --- Message if everything went well
########################################################################################################################
print('\n')
print('OK! No error (lambda-functions) vs. (raw vec.-mat.-notation) was found.')
exit(0)
