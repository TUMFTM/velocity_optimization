import csv
import json
import numpy as np
import configparser


class CalcObjective:

    __slots__ = ('csv_name',
                 'params_path',
                 'log_lines',
                 'J_vvmax',
                 'J_eps_tre_lin',
                 'J_eps_tre_quad',
                 'J_jerk',
                 'qp_iter_last_arr',
                 'qp_status_last_arr',
                 'dt_sqp_arr',
                 'params')

    def __init__(self,
                 csv_name: str,
                 log_lines: int,
                 sid: str,
                 params_path: str):
        """
        Python version: 3.5
        Created by: Thomas Herrmann (thomas.herrmann@tum.de)
        Created on: 01.11.2019

        Documentation: Class to read available velocity-optimization-logs and re-calculate the objective function J
        for the entirely logged session

        Inputs:
        csv_name: Path and file name to log-file
        log_lines: Number of lines in log file that build one data-block
        sid: ID 'PerfSQP' or 'EmergSQP' of SQP optimization object
        params_path: absolute path to folder containing config file .ini
        """

        self.csv_name = csv_name

        self.params_path = params_path

        # Parameter for repeating log variables after number of lines
        self.log_lines = log_lines

        self.J_vvmax = []
        self.J_eps_tre_lin = []
        self.J_eps_tre_quad = []
        self.J_jerk = []
        self.qp_iter_last_arr = []
        self.qp_status_last_arr = []
        self.dt_sqp_arr = []

        self.params = dict()

        self.read_params(sid=sid)

    def read_params(self,
                    sid: str) -> dict:
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.11.2019

        Documentation: Read parameters of SQP velocity optimizer

        Inputs:
        sid: ID 'PerfSQP' or 'EmergSQP' of SQP optimization object

        Outputs:
        params: QP solver parameters
        """

        sqp_config = configparser.ConfigParser()
        if not sqp_config.read(self.params_path + 'sqp_config.ini'):
            raise ValueError('Specified cost config file does not exist or is empty!')

        if sid == 'PerfSQP':
            traj_type = 'SOLVER_PERFORMANCE'
        elif sid == 'EmergSQP':
            traj_type = 'SOLVER_EMERGENCY'
        else:
            traj_type = None

        self.params = {'m_t_': sqp_config.getfloat('VEHICLE', 'mass_t'),
                       'Pmax_kW_': sqp_config.getfloat('VEHICLE', 'P_max_kW'),
                       'Fmax_kN_': sqp_config.getfloat('VEHICLE', 'F_max_kN'),
                       'Fmin_kN_': sqp_config.getfloat('VEHICLE', 'F_min_kN'),
                       'axmax_mps2_': sqp_config.getfloat('VEHICLE', 'ax_max_mps2'),
                       'aymax_mps2_': sqp_config.getfloat('VEHICLE', 'ay_max_mps2'),
                       'dF_kN_pos_': sqp_config.getfloat('SOLVER_GENERAL', 'dF_kN_pos'),
                       'dF_kN_neg_': sqp_config.getfloat('SOLVER_GENERAL', 'dF_kN_neg'),
                       'Fini_tol_': sqp_config.getfloat(traj_type, 'F_ini_tol'),
                       'c_res_': sqp_config.getfloat('VEHICLE', 'c_res'),
                       'vmin_mps_': sqp_config.getfloat('SOLVER_GENERAL', 'v_min_mps'),
                       's_v_t_lim_': sqp_config.getfloat(traj_type, 'slack_var_tire_lim'),
                       's_v_t_unit_': sqp_config.getfloat('SOLVER_GENERAL', 'slack_var_tire_unit'),
                       'dvdv_w_': sqp_config.getfloat(traj_type, 'penalty_jerk'),
                       'tre_cst_w_': sqp_config.getfloat(traj_type, 'w_tre_constraint'),
                       's_tre_w_lin_': sqp_config.getfloat(traj_type, 'penalty_slack_tire_lin'),
                       's_tre_w_quad_': sqp_config.getfloat(traj_type, 'penalty_slack_tire_quad')
                       }

        return self.params

    def calc_objective(self):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.11.2019

        Documentation: Re-calculates the objective function J from logged data as well as it stores the status,
        iteration number and QP runtime of the last QP within the SQP
        """

        with open(self.csv_name) as csvfile:

            # Open CSV file
            rcsv = csv.reader(csvfile,
                              delimiter=';',
                              lineterminator='\n')

            for idx, line in enumerate(rcsv):

                # --- Get v_max, iterations, status, QP runtime
                if np.mod(idx, self.log_lines) == 0:

                    v_max_mps = np.array(json.loads(line[7]))
                    iters = json.loads(line[11])
                    status = json.loads(line[12])
                    dt_sqp_ms = json.loads(line[13])

                # --- Get optimized velocity
                if np.mod(idx, self.log_lines) == 1:
                    v_mps = (json.loads(line[0]))
                    self.J_vvmax.append(np.sum((v_mps - v_max_mps) ** 2))

                # --- Get optimized slack values
                if np.mod(idx, self.log_lines) == 2:

                    slacks = np.array((json.loads(line[0])))

                    self.J_eps_tre_lin.append(
                        self.params['s_tre_w_lin_']
                        * np.sum(self.params['s_v_t_unit_'] * slacks)
                    )

                    self.J_eps_tre_quad.append(
                        self.params['s_tre_w_quad_']
                        * np.sum(np.square(self.params['s_v_t_unit_'] * slacks))
                    )

                    self.J_jerk.append(
                        self.params['dvdv_w_']
                        * np.sum(
                            np.square(np.diff(np.diff(v_mps)))
                        )
                    )

                    self.qp_iter_last_arr.append(iters)
                    self.qp_status_last_arr.append(status)
                    self.dt_sqp_arr.append(dt_sqp_ms)


if __name__ == '__main__':
    pass
