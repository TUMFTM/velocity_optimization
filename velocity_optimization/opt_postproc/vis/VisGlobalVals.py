from matplotlib import pyplot as plt
import numpy as np
import linecache
import json
try:
    import tikzplotlib
except ImportError:
    print('Warning: No module tikzplotlib found. Not necessary on car but for development.')
# from velocity_optimization.opt_postproc.src.read_ltpl_raw import read_ltpl_raw

# Line width
LW = 1.5

# The higher the less bins are plotted
BIN_WIDTH = 20


class VisVP_GlobalVals:

    __slots__ = ()

    def __init__(self,
                 vis_main,
                 csv_name: str,
                 csv_name_ltpl: str,
                 log_lines: int,
                 row_count: int,
                 glob_lim: int):

        """
        Python version: 3.5
        Created by: Thomas Herrmann (thomas.herrmann@tum.de)
        Created on: 01.02.2020

        Documentation: Class for the visualization of global values for an entirely logged race session.

        Inputs:
        vis_main: Main object for logging data visualization
        csv_name: path + name of csv-file to logged data
        csv_name_ltpl: raw local trajectory planner raw file
        log_line: number of lines building one data block in logged data
        row_count: number of rows in log file
        glob_lim: limit of lines to be read from logs
        """

        self.extract_vals(vis_main=vis_main,
                          csv_name=csv_name,
                          csv_name_ltpl=csv_name_ltpl,
                          log_lines=log_lines,
                          row_count=row_count,
                          glob_lim=glob_lim)

    def extract_vals(self,
                     vis_main,
                     csv_name: str,
                     csv_name_ltpl: str,
                     log_lines: int,
                     row_count: int,
                     glob_lim: int):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.02.2020

        Documentation: Visualization of global values for an entirely logged race session.

        Inputs:
        vis_main: Main object for logging data visualization
        csv_name: path + name of csv-file to logged data
        csv_name_ltpl: raw local trajectory planner raw file
        log_line: number of lines building one data block in logged data
        row_count: number of rows in log file
        glob_lim: limit of lines to be read from logs
        """

        # --- Initialize empty lists
        s_glob_m = []
        v_glob_mps = []
        P_glob_kW = []
        F_glob_kN = []

        # --- Read from file
        for idx in range(1, row_count):

            row_lc = linecache.getline(csv_name, idx)
            # Data format conversion
            row_lc = row_lc[:-1].rsplit(';')

            if idx == glob_lim:
                break
            if np.mod(idx, log_lines) == 1:

                # Status
                print(str(idx) + "  /  " + str(row_count))

                # Update of constraints
                vis_main.vis_log(idx - 1)

                s_m = json.loads(row_lc[1])
                s_glob_m.append(s_m)

                v_mps = json.loads(row_lc[6])
                v_glob_mps.append(v_mps)

                # Check for variable power
                if vis_main.Pmax_kW.size == 1:
                    P_glob_kW.append(float(vis_main.velqp.P_cst[0]) + float(vis_main.P_max))
                else:
                    P_glob_kW.append(float(vis_main.velqp.P_cst[0]) + float(vis_main.P_max[0]))

                F_glob_kN.append(float(vis_main.velqp.F_cst[0]) + vis_main.velqp.sym_sc_['Fmax_kN_'])

        ################################################################################################################
        # Postprocessing
        ################################################################################################################

        # --- Retrieve LTPL raw logs
        '''
        s_glob_m_ltpl, \
            v_mps_ltpl, \
            ax_mps2_ltpl, \
            ay_mps2_ltpl = read_ltpl_raw(csv_name=csv_name_ltpl)

        F_kN_ltpl = ax_mps2_ltpl * vis_main.velqp.sym_sc_['m_t_']
        F_kN_ltpl[F_kN_ltpl < 0] = 0
        E_glob_kJ_ltpl = np.cumsum(np.diff(s_glob_m_ltpl[4332:6382]) * F_kN_ltpl[4332:6381])
        '''

        # --- Match entries to best fitting global s coordinate

        s_glob_m_filt = []
        v_glob_mps_filt = []
        P_glob_kW_filt = []
        F_glob_kN_filt = []
        for i in range(0, len(s_glob_m) - 2):
            s_tmp = s_glob_m[i]
            v_tmp = v_glob_mps[i]
            P_tmp = P_glob_kW[i]
            F_tmp = F_glob_kN[i]
            if s_tmp == s_glob_m[i + 1]:
                pass
            else:
                s_glob_m_filt.append(s_tmp)
                v_glob_mps_filt.append(v_tmp)
                P_glob_kW_filt.append(P_tmp)
                F_glob_kN_filt.append(F_tmp)

        # --- Overwrite negative values (no recuperation)
        F_glob_kN_filt = np.array(F_glob_kN_filt)
        F_glob_kN_filt[F_glob_kN_filt < 0] = 0
        P_glob_kW_filt_pos = np.array(P_glob_kW_filt)
        P_glob_kW_filt_pos[P_glob_kW_filt_pos < 0] = 0

        dt_glob_s = np.diff(s_glob_m_filt) / v_glob_mps_filt[:-1]

        E_glob_kJ_filt = np.cumsum(dt_glob_s * P_glob_kW_filt_pos[:-1])
        E_glob_kJ_filt = np.cumsum(np.diff(s_glob_m_filt) * F_glob_kN_filt[0:-1])

        # --- Power
        plt.figure()
        plt.plot(s_glob_m_filt, P_glob_kW_filt)
        plt.xlabel(r'$s$' + ' in ' + r'm')
        plt.ylabel(r'$P$' + ' in ' + r'kW')
        tikzplotlib.save("power.tex")

        # --- Energy
        plt.figure()
        plt.plot(s_glob_m_filt[1:], E_glob_kJ_filt / 3600)
        plt.xlabel(r'$s$' + ' in ' + r'm')
        plt.ylabel(r'$E$' + ' in ' + r'kWh')
        tikzplotlib.save("energy.tex")
        plt.show()


if __name__ == '__main__':
    pass
