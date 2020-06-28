import numpy as np
import os
import logging
import sympy
from velocity_optimization.src.SymQP import SymQP


def log_sparsity(symbolics: sympy.ImmutableDenseMatrix,
                 name: str,
                 logger: logging.Logger):
    """
    Python version: 3.5
    Created by: Thomas Herrmann (thomas.herrmann@tum.de)
    Created on: 01.03.2020

    Documentation: Writes sparsity pattern to file.

    Inputs:
    symbolics: symbolic matrices of the constraint expressions
    name: name of constraint
    logger: logger to be used to write pattern to external file
    """

    # --- Get sparsity pattern of constraints in rows and columns
    rc = np.nonzero(symbolics)
    r = rc[0]
    c = rc[1]

    logger.debug('%s', '[' + name + ']')

    logger.debug('%s', 'r=' + str(r.tolist()))
    logger.debug('%s', 'c=' + str(c.tolist()))


def calc_sparsity(params_path: str,
                  logging_path: str,
                  m_perf: int = 115,
                  m_emerg: int = 50):
    """
    Python version: 3.5
    Created by: Thomas Herrmann (thomas.herrmann@tum.de)
    Created on: 01.03.2020

    Documentation: Derives the sparsity pattern of the constraint matrix A that the OSQP solver needs.
    The pattern is derived from the symbolic expressions of the linearized constraints. The pattern is used
    online to directly fill the sparse CSC A-matrix for faster and more stable performance.

    Inputs:
    m_perf: number of velocity points in performance profile
    m_emerg: number of velocity points in emergency profile
    """

    ####################################################################################################################
    # --- Calculate sparsity pattern of Performance SQP
    ####################################################################################################################

    m = m_perf
    sid = 'PerfSQP'

    # --- Create logger for results
    logger = logging.getLogger('sqp_logger_perf')
    logger.setLevel(logging.DEBUG)

    # create logs folder
    logging_path += '/sparsity'
    os.makedirs(logging_path, exist_ok=True)

    fh = logging.FileHandler(logging_path + '/sqp_sparsity_'
                             + sid
                             + str(m)
                             + '.ini')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # --- Create SymQP-instance
    symqp = SymQP(m=m,
                  sid=sid,
                  params_path=params_path)

    log_sparsity(symqp.F_ini_cst_jac[:, 1:], 'symqp.F_sym_ini_cst_jac_:_1:', logger)
    log_sparsity(symqp.F_cst_jac[1:, 1:], 'symqp.F_sym_cst_jac_1:_1:', logger)
    log_sparsity(symqp.P_cst_jac[:, 1:], 'symqp.P_sym_cst_jac_:_1:', logger)
    log_sparsity(symqp.Tre_cst1_jac[:, 1:], 'symqp.Tre_sym_cst1_jac_:_1:', logger)
    log_sparsity(symqp.Tre_cst2_jac[:, 1:], 'symqp.Tre_sym_cst2_jac_:_1:', logger)
    log_sparsity(symqp.Tre_cst3_jac[:, 1:], 'symqp.Tre_sym_cst3_jac_:_1:', logger)
    log_sparsity(symqp.Tre_cst4_jac[:, 1:], 'symqp.Tre_sym_cst4_jac_:_1:', logger)

    ####################################################################################################################
    # --- Calculate sparsity pattern of Emergency SQP
    ####################################################################################################################
    m = m_emerg
    sid = 'EmergSQP'

    # --- Create SymQP-instance
    symqp = SymQP(m=m,
                  sid=sid,
                  params_path=params_path)

    # --- Create logger for results
    logger = logging.getLogger('sqp_logger_emerg')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(logging_path + '/sqp_sparsity_'
                             + sid
                             + str(m)
                             + '.ini')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    log_sparsity(symqp.F_cst_jac[:, 1:], 'symqp.F_sym_cst_jac_:_1:', logger)
    log_sparsity(symqp.P_cst_jac[:, 1:], 'symqp.P_sym_cst_jac_:_1:', logger)
    log_sparsity(symqp.Tre_cst1_jac[:, 1:], 'symqp.Tre_sym_cst1_jac_:_1:', logger)
    log_sparsity(symqp.Tre_cst2_jac[:, 1:], 'symqp.Tre_sym_cst2_jac_:_1:', logger)
    log_sparsity(symqp.Tre_cst3_jac[:, 1:], 'symqp.Tre_sym_cst3_jac_:_1:', logger)
    log_sparsity(symqp.Tre_cst4_jac[:, 1:], 'symqp.Tre_sym_cst4_jac_:_1:', logger)


if __name__ == '__main__':

    params_path_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/params/'
    logging_path_ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/logs'

    calc_sparsity(params_path=params_path_,
                  logging_path=logging_path_,
                  m_perf=5,
                  m_emerg=5)
