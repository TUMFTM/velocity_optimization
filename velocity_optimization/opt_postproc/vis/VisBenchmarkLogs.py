import csv
import json
import numpy as np
import datetime
import time
from matplotlib import pyplot as plt
import linecache
from velocity_optimization.src.VelQP import VelQP
from velocity_optimization.opt_postproc.src.VOptIPOPT import VOptIPOPT
from velocity_optimization.opt_postproc.src.VOptQPOASES import VOpt_qpOASES
from velocity_optimization.opt_postproc.src.online_qp_postproc import online_qp_postproc
from velocity_optimization.opt_postproc.src.CalcObjective import CalcObjective
from velocity_optimization.opt_postproc.vis.VisGUI import VisVP_Logs_GUI
from velocity_optimization.opt_postproc.vis.VisObjectiveStatus import VisVP_ObjStatus
from velocity_optimization.opt_postproc.vis.VisRuntime import VisVP_Runtime
from velocity_optimization.opt_postproc.vis.VisGlobalVals import VisVP_GlobalVals
try:
    import tikzplotlib
except ImportError:
    print('Warning: No module tikzplotlib found. Not necessary on car but for development.')

# Font sizes
SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

# Line Width
LW = 2


class VisVP_Logs:

    __slots__ = ('csv_name',
                 'csv_name_ltpl',
                 'vis_options',
                 'sol_options',
                 'sol_dict',
                 'm',
                 'sid',
                 'log_lines',
                 'row_count',
                 'calc_objective',
                 'runtimes',
                 'vis_gui',
                 'velqp',
                 'velqp_bench',
                 'vp_ipopt',
                 'vp_qpOASES',
                 'b_vis_triggered',
                 'dt_ipopt_arr',
                 'dt_sqp_qpoases_arr',
                 'P_max')

    def __init__(self,
                 csv_name: str,
                 csv_name_ltpl: str,
                 params_path: str,
                 input_path: str,
                 vis_options: dict,
                 sol_options: dict,
                 m: int,
                 sid: str = 'PerfSQP',
                 log_lines: int = 4):

        """
        Python version: 3.5
        Created by: Thomas Herrmann (thomas.herrmann@tum.de)
        Created on: 01.02.2020
        Modified by Tobias Klotz

        Documentation: Class doing the main handling of the visualization of the SQP velocity planner logging data.

        Inputs:
        csv_name: log data path & file
        csv_name_ltpl: raw lof file of ltpl module
        params_path: absolute path to folder containing config file .ini
        input_path: absolute path to folder containing variable vehicle and track information
        vis_options: user specified visualization options of the debugging tool
        m: number of velocity optimization variables
        sid: ID of the velocity planner 'PerfSQP' or 'EmergSQP'
        log_lines: number of lines in log files spanning one cohesive data block
        """

        self.csv_name = csv_name
        self.csv_name_ltpl = csv_name_ltpl
        self.vis_options = vis_options
        self.sol_options = sol_options
        self.m = m
        self.sid = sid

        self.log_lines = log_lines
        with open(csv_name) as csvfile:
            self.row_count = sum(1 for row in
                                 csv.reader(csvfile,
                                            delimiter=';',
                                            lineterminator='\n'))

        # --- Create objective calculation object
        self.calc_objective = CalcObjective(csv_name=csv_name,
                                            log_lines=log_lines,
                                            sid=sid,
                                            params_path=params_path)
        self.calc_objective.calc_objective()

        # --- Create runtime histogram object
        self.runtimes = VisVP_Runtime()

        # --- Create visualization GUI object
        self.vis_gui = VisVP_Logs_GUI(vis_handler=self,
                                      m=m,
                                      vis_options=vis_options,
                                      params_path=params_path,
                                      sol_options=sol_options)

        # --- Create OSQP solver object
        self.velqp = VelQP(m=m,
                           sid=sid,
                           params_path=params_path,
                           input_path=input_path)

        self.velqp_bench = VelQP(m=m,
                                 sid=sid,
                                 params_path=params_path,
                                 input_path=input_path)

        # --- Create Solver Objects
        for key, value in self.sol_options.items():

            # --- Create IPOPT solver object
            if self.sol_options[key]['Solver'] == "IPOPT":
                self.sol_options[key].update({'Create_Solver': VOptIPOPT(m=m,
                                                                         sid=sid,
                                                                         params_path=params_path,
                                                                         b_warm=False,
                                                                         vis_options=vis_options,
                                                                         sol_dict=sol_options,
                                                                         key=key)})

            # --- Create qpOASES solver object
            if self.sol_options[key]['Solver'] == "qpOASES":
                self.sol_options[key].update({'Create_Solver': VOpt_qpOASES(Hm=self.velqp_bench.J_Hess[1:, 1:],
                                                                            Am=self.velqp_bench.Am)})

        # For movie tool
        self.b_vis_triggered = False

        # Runtime arrays
        self.dt_sqp_qpoases_arr = []

        # Runtime and solver status arrays
        for key, value in self.sol_options.items():
            self.sol_options[key]['Time'] = []
            self.sol_options[key]['SolStatus'] = []

        # Store max. power values
        self.P_max = None

        # --- Do some plots extra to main debug window
        self.pre_debug_plots()

    def pre_debug_plots(self):

        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.02.2020

        Documentation: Does extra plots before main debug window opens. These plots contain QP status,
        objective term values, global values from entirely logged race session.
        """

        # --- Visualize objective and SQP status
        VisVP_ObjStatus(calc_objective=self.calc_objective,
                        vis_options=self.vis_options)

        # --- Visualize SQP (OSQP) runtime
        if self.vis_options['b_calc_time_plot']:
            self.runtimes.plot_hist(name_solver='OSQP',
                                    dt_solver=self.calc_objective.dt_sqp_arr,
                                    vis_options=self.vis_options)

        # --- Visualization of global values (energy, power)
        if self.vis_options['b_global_plot']:
            self.glob_val_vis = VisVP_GlobalVals(vis_main=self,
                                                 csv_name=self.csv_name,
                                                 csv_name_ltpl=self.csv_name_ltpl,
                                                 log_lines=self.log_lines,
                                                 row_count=self.row_count,
                                                 glob_lim=self.vis_options['glob_lim'])

    def vis_log(self, idx: int):

        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.02.2020

        Documentation: Reads data from logging files and updates the main debug window.

        Inputs:
        idx: the starting line-number of the data block that should be read from the log-file
        """

        # Modify user's idx to have multiple of log_lines
        print(idx)
        if idx != 0 and np.mod(idx, self.log_lines) != 0:
            idx = int(idx + (self.log_lines - np.mod(idx, self.log_lines)))
        elif idx == self.row_count:
            pass

        # don't extract entries corresponding to index out of bounds
        if idx >= self.row_count:
            idx -= self.log_lines

        # Choose your own starting Index for a single plot
        if self.vis_options['b_idx'] > 0:
            number = self.vis_options['b_idx']
            idx = 4 * number

        # --- Read specific lines in logs
        row_lc1 = self.read_log_file(self.csv_name, idx)
        row_lc2 = self.read_log_file(self.csv_name, idx + 1)
        row_lc3 = self.read_log_file(self.csv_name, idx + 2)
        row_lc4 = self.read_log_file(self.csv_name, idx + 3)

        if (row_lc1 != []) and (row_lc2 != []) and (row_lc3 != []) and (row_lc4 != []):

            # Timestamp
            ts = row_lc1[0]  # datetime.datetime(2012,4,1,0,0).timestamp()
            # --- Get data from file-name: year, month, day
            ts_prefix = self.csv_name.rsplit('/')[-1].rsplit('_')[2] + '-' \
                + self.csv_name.rsplit('/')[-1].rsplit('_')[3] + '-' \
                + self.csv_name.rsplit('/')[-1].rsplit('_')[4] + '-'
            ts = ts_prefix + ts

            # Global s coordinate [m]
            s_glob = json.loads(row_lc1[1])
            # Initial velocity guess [m/s]
            x0_v = np.array(json.loads(row_lc1[2]))
            # Initial slack variable value guess [-]
            x0_s_t = np.array(json.loads(row_lc1[3]))
            # kappa profile of path [rad/m]
            kappa = np.array(json.loads(row_lc1[4]))
            # Step size of given path [m]
            delta_s = np.array(json.loads(row_lc1[5]))
            # Initial velocity hard constraint [m/s]
            v_ini = json.loads(row_lc1[6])
            # Max. allowed velocity in optimization horizon [m/s]
            v_max = np.array(json.loads(row_lc1[7]))
            # End velocity in optimization horizon [m/s]
            v_end = json.loads(row_lc1[8])
            # Initial force constraint [kN]
            F_ini = json.loads(row_lc1[9])
            # Max. allowed power [kW]
            P_max = np.array(json.loads(row_lc1[10]))
            self.P_max = P_max

            # Iteration number of last QP in velocity SQP [-]
            qp_iter = json.loads(row_lc1[11])
            # Status of last QP in velocity SQP [-]
            qp_status = json.loads(row_lc1[12])
            # Total optimization time of velocity SQP [ms]
            dt_sqp = json.loads(row_lc1[13])

            # By OSQP optimized velocity profile [m/s]
            v_op_osqp = json.loads(row_lc2[0])

            # By OSQP optimized slack variable values [-]
            s_t_op_osqp = json.loads(row_lc3[0])

            # Max. allowed longitudinal acceleration [m/s^2]
            ax_max = json.loads(row_lc4[0])
            # Max. allowed lateral acceleration in optimization horizon [m/s^2]
            ay_max = json.loads(row_lc4[1])

            # --- Update constraints in QP
            if np.size(ax_max) == 1:
                ax_max_new = None
                ax_max_new_ipopt = None
                ay_max_new = None
                ay_max_new_ipopt = None
            else:
                ax_max_new = ax_max
                ax_max_new_ipopt = ax_max_new[:-1]
                ay_max_new = ay_max
                ay_max_new_ipopt = ay_max_new[:-1]

            # --- Update OSQP expressions
            self.velqp.osqp_update_online(x0_v=np.array(v_op_osqp),
                                          x0_s_t=np.array(s_t_op_osqp),
                                          v_ini=v_ini,
                                          v_max=v_max,
                                          v_end=v_end,
                                          F_ini=F_ini,
                                          P_max=P_max,
                                          kappa=kappa,
                                          delta_s=delta_s,
                                          ax_max=ax_max_new,
                                          ay_max=ay_max_new)

            ############################################################################################################
            # --- Rerun OSQP from log-data
            ############################################################################################################

            if self.vis_options['b_run_OSQP']:

                # --- Rerun OSQP solver
                sol = self.velqp.osqp_solve()[0]

                # --- Transform OSQP solution
                v_op_rerun = np.insert(sol[0:self.velqp.m - 1], 0, 0) + x0_v

                # s_t_op_rerun = sol[self.velqp.m - 1:] + x0_s_t

                print("OSQP rerun v_mps:\n", v_op_rerun)

            ############################################################################################################
            # --- Calculate Solver-solution (Dictionary)
            ############################################################################################################
            for key, value in self.sol_options.items():

                ########################################################################################################
                # --- Calculate IPOPT-solution
                ########################################################################################################

                if self.sol_options[key]['Solver'] == "IPOPT":
                    sol_ipopt, \
                        param_vec, \
                        t_op_ipopt, \
                        sol_status_ipopt = self.sol_options[key]['Create_Solver'].calc_v_ipopt(v_ini=v_ini,
                                                                                               kappa=kappa,
                                                                                               delta_s=delta_s,
                                                                                               v_max=v_max,
                                                                                               F_ini=F_ini,
                                                                                               v_end=v_end,
                                                                                               P_max=P_max,
                                                                                               x0_v=x0_v,
                                                                                               x0_s_t=x0_s_t,
                                                                                               ax_max=ax_max_new_ipopt,
                                                                                               ay_max=ay_max_new_ipopt)

                    v_op_ipopt, \
                        s_t_op_ipopt, \
                        F_op_ipopt, \
                        P_op_ipopt, \
                        ax_op_ipopt, \
                        ay_op_ipopt, \
                        F_xzf_op_ipopt, \
                        F_yzf_op_ipopt, \
                        F_xzr_op_ipopt, \
                        F_yzr_op_ipopt, \
                        F_xzfl_op_ipopt, \
                        F_xzfr_op_ipopt, \
                        F_yzfl_op_ipopt, \
                        F_yzfr_op_ipopt, \
                        F_xzrl_op_ipopt, \
                        F_xzrr_op_ipopt, \
                        F_yzrl_op_ipopt, \
                        F_yzrr_op_ipopt, \
                        = self.sol_options[key]['Create_Solver'].transform_sol(sol=sol_ipopt,
                                                                               param_vec_=param_vec,
                                                                               vis_options=self.vis_options)

                    self.sol_options[key].update({'Velocity': v_op_ipopt})
                    self.sol_options[key].update({'Slack_Sol': s_t_op_ipopt})
                    self.sol_options[key].update({'Force': F_op_ipopt})
                    self.sol_options[key].update({'Power': P_op_ipopt})
                    self.sol_options[key].update({'Acc_x': ax_op_ipopt})
                    self.sol_options[key].update({'Acc_y': ay_op_ipopt})
                    self.sol_options[key].update({'F_xzf': F_xzf_op_ipopt})
                    self.sol_options[key].update({'F_yzf': F_yzf_op_ipopt})
                    self.sol_options[key].update({'F_xzr': F_xzr_op_ipopt})
                    self.sol_options[key].update({'F_yzr': F_yzr_op_ipopt})
                    self.sol_options[key].update({'F_xzfl': F_xzfl_op_ipopt})
                    self.sol_options[key].update({'F_xzfr': F_xzfr_op_ipopt})
                    self.sol_options[key].update({'F_yzfl': F_yzfl_op_ipopt})
                    self.sol_options[key].update({'F_yzfr': F_yzfr_op_ipopt})
                    self.sol_options[key].update({'F_xzrl': F_xzrl_op_ipopt})
                    self.sol_options[key].update({'F_xzrr': F_xzrr_op_ipopt})
                    self.sol_options[key].update({'F_yzrl': F_yzrl_op_ipopt})
                    self.sol_options[key].update({'F_yzrr': F_yzrr_op_ipopt})
                    self.sol_options[key]['Time'].append(t_op_ipopt)
                    self.sol_options[key]['SolStatus'].append(sol_status_ipopt)

                ########################################################################################################
                # --- Calculate qpOASES-solution
                ########################################################################################################

                if self.sol_options[key]['Solver'] == "qpOASES":

                    # qpOASES velocity [m/s]
                    v_op_qpoases = None
                    # qpOASES force [kN]
                    F_op_qpoases = None
                    # qpOASES tire slacks [-]
                    eps_op_qpoases = None

                    t_start = time.perf_counter()

                    v_op_qpoases, \
                        eps_op_qpoases, \
                        F_op_qpoases = \
                        online_qp_postproc(velqp=self.velqp_bench,
                                           vp_qpOASES=self.sol_options[key]['Create_Solver'],
                                           v_ini=v_ini,
                                           kappa=kappa,
                                           delta_s=delta_s,
                                           x0_v=x0_v,
                                           x0_s_t=x0_s_t,
                                           v_max=v_max,
                                           v_end=v_end,
                                           F_ini=F_ini,
                                           ax_max=ax_max_new,
                                           ay_max=ay_max_new,
                                           P_max=P_max)

                    # Overall SQP runtime using qpOASES [ms]
                    dt_sqp_qpoases = (time.perf_counter() - t_start) * 1000

                    # Store qpOASES runtime
                    print('Overall qpOASES runtime:', dt_sqp_qpoases)

                    P_op_qpoases = F_op_qpoases[0:self.m - 1] * v_op_qpoases[0:self.m - 1]

                    ay_op_qpoases = kappa[0:self.velqp.sqp_stgs['m'] - 1] \
                        * np.square(v_op_qpoases[0:self.velqp.sqp_stgs['m'] - 1])

                    ax_op_qpoases = (np.square(v_op_qpoases[1:self.velqp.sqp_stgs['m']]) - np.square(
                        v_op_qpoases[0:self.velqp.sqp_stgs['m'] - 1])) / (2 * np.array(delta_s)) \
                        + (self.velqp.sym_sc_['c_res_']
                           * np.square(v_op_qpoases[0:self.velqp.sqp_stgs['m'] - 1])) /\
                          (1000 * self.velqp.sym_sc_['m_t_'])

                    self.sol_options[key].update({'Velocity': v_op_qpoases})
                    self.sol_options[key].update({'Slack_Sol': eps_op_qpoases})
                    self.sol_options[key].update({'Force': F_op_qpoases})
                    self.sol_options[key].update({'Power': P_op_qpoases})
                    self.sol_options[key].update({'Acc_x': ax_op_qpoases})
                    self.sol_options[key].update({'Acc_y': ay_op_qpoases})
                    self.sol_options[key]['Time'].append(dt_sqp_qpoases)
                    # self.sol_options[key]['SolStatus'].append(sol_status_qpoases)

            if self.vis_options['b_immediate_plot_update']:

                # --- SQP Status update
                self.vis_gui.text_status.\
                    set_text(r'$s_\mathrm{glob}$: ' + str('%.2f' % s_glob) + ' m' + r'   $t$: '
                             + str(3600 + time.mktime(datetime.datetime.strptime(ts, "%Y-%m-%d-%H:%M:%S.%f").timetuple()
                                                      )) + ' s'
                             + r'   $\mathrm{QP}_\mathrm{status}$: ' + str(qp_status)
                             + r'   $\mathrm{QP}_\mathrm{iter}$: ' + str(qp_iter)
                             + r'   $\mathrm{SQP}_\mathrm{\Delta t}$: ' + str(dt_sqp) + ' ms')

                # --- Velocity update
                self.vis_gui.p1_1.set_ydata(v_op_osqp[0:self.velqp.m])
                self.vis_gui.p1_2.set_ydata(v_max)
                self.vis_gui.p1_3.set_ydata(v_end)
                self.vis_gui.p1_4.set_ydata(x0_v)

                # debugging solver updates
                for key, value in self.sol_options.items():
                    self.vis_gui.vel_dict[key].set_ydata(self.sol_options[key]['Velocity'])

                # --- Force update
                self.vis_gui.p3_1.set_ydata(self.velqp.F_cst + self.velqp.sym_sc_['Fmax_kN_'])
                self.vis_gui.p3_2.set_ydata([F_ini,
                                             F_ini + self.velqp.sym_sc_['Fini_tol_'],
                                             F_ini - self.velqp.sym_sc_['Fini_tol_']])

                for key, value in self.sol_options.items():
                    self.vis_gui.F_dict[key].set_ydata(self.sol_options[key]['Force'])

                # --- Delta Force update
                self.vis_gui.p4_1.set_ydata(self.velqp.dF_cst + self.velqp.sym_sc_['dF_kN_pos_'])

                # --- Power
                if self.velqp.sqp_stgs['b_var_power']:
                    self.vis_gui.p5_1.set_ydata(self.velqp.P_cst + P_max.reshape((self.velqp.m - 1, 1)))
                    self.vis_gui.p5_2.set_ydata(P_max)
                    for key, value in self.sol_options.items():
                        self.vis_gui.P_dict[key].set_ydata(np.around(self.sol_options[key]['Power'], 2))
                else:
                    self.vis_gui.p5_1.set_ydata(self.velqp.P_cst + self.velqp.sym_sc_['Pmax_kW_'])
                    self.vis_gui.p5_2.set_ydata(self.velqp.sym_sc_['Pmax_kW_'] * np.ones((self.m - 1, 1)))
                    for key, value in self.sol_options.items():
                        self.vis_gui.P_dict[key].set_ydata(np.array(self.sol_options[key]['Power'] * 1.0))

                # --- Tire usage update
                ay_mps2 = kappa[0:self.velqp.sqp_stgs['m'] - 1] * \
                    np.square(v_op_osqp[0:self.velqp.sqp_stgs['m'] - 1])

                ax_mps2 = (np.square(v_op_osqp[1:self.velqp.sqp_stgs['m']]) - np.square(
                    v_op_osqp[0:self.velqp.sqp_stgs['m'] - 1])) / (2 * np.array(delta_s)) + \
                    (self.velqp.sym_sc_['c_res_'] * np.square(v_op_osqp[0:self.velqp.sqp_stgs['m'] - 1])) / \
                    (1000 * self.velqp.sym_sc_['m_t_'])

                self.vis_gui.p6_1.set_xdata(ay_mps2)
                self.vis_gui.p6_1.set_ydata(ax_mps2)
                for key, value in self.sol_options.items():
                    self.vis_gui.a_dict[key].set_xdata(self.sol_options[key]['Acc_y'])
                    self.vis_gui.a_dict[key].set_ydata(self.sol_options[key]['Acc_x'])

                # Case of variable friction limits
                if np.size(ax_max) > 1:
                    axmax_max_plot = np.max(ax_max)
                    axmax_min_plot = np.min(ax_max)
                    aymax_max_plot = np.max(ay_max)
                    aymax_min_plot = np.min(ay_max)

                    self.vis_gui.p6_2.set_xdata([aymax_max_plot, 0, - aymax_max_plot, 0, aymax_max_plot])
                    self.vis_gui.p6_2.set_ydata([0, axmax_max_plot, 0, - axmax_max_plot, 0])
                    self.vis_gui.p6_3.set_xdata([aymax_min_plot, 0, - aymax_min_plot, 0, aymax_min_plot])
                    self.vis_gui.p6_3.set_ydata([0, axmax_min_plot, 0, - axmax_min_plot, 0])

                # --- Front Axle Tire usage update
                for key, value in self.sol_options.items():
                    if self.sol_options[key]['Model'] == "DM" and self.sol_options[key]['Solver'] == "IPOPT":
                        self.vis_gui.F_f_dict[key].set_xdata(self.sol_options[key]['F_yzf'])
                        self.vis_gui.F_f_dict[key].set_ydata(self.sol_options[key]['F_xzf'])
                    elif self.sol_options[key]['Model'] == "FW" and self.sol_options[key]['Solver'] == "IPOPT":
                        self.vis_gui.F_fl_dict[key].set_xdata(self.sol_options[key]['F_yzfl'])
                        self.vis_gui.F_fl_dict[key].set_ydata(self.sol_options[key]['F_xzfl'])
                        self.vis_gui.F_fr_dict[key].set_xdata(self.sol_options[key]['F_yzfr'])
                        self.vis_gui.F_fr_dict[key].set_ydata(self.sol_options[key]['F_xzfr'])

                # --- Rear Axle Tire usage update
                for key, value in self.sol_options.items():
                    if self.sol_options[key]['Model'] == "DM" and self.sol_options[key]['Solver'] == "IPOPT":
                        self.vis_gui.F_r_dict[key].set_xdata(self.sol_options[key]['F_yzr'])
                        self.vis_gui.F_r_dict[key].set_ydata(self.sol_options[key]['F_xzr'])
                    elif self.sol_options[key]['Model'] == "FW" and self.sol_options[key]['Solver'] == "IPOPT":
                        self.vis_gui.F_rl_dict[key].set_xdata(self.sol_options[key]['F_yzrl'])
                        self.vis_gui.F_rl_dict[key].set_ydata(self.sol_options[key]['F_xzrl'])
                        self.vis_gui.F_rr_dict[key].set_xdata(self.sol_options[key]['F_yzrr'])
                        self.vis_gui.F_rr_dict[key].set_ydata(self.sol_options[key]['F_xzrr'])

                # --- Slack update
                self.vis_gui.p7_1.set_ydata(s_t_op_osqp)
                self.vis_gui.p7_2.set_ydata(x0_s_t)
                self.vis_gui.p7_3.set_ydata([self.velqp.sym_sc_['s_v_t_lim_']] * len(s_t_op_osqp))
                for key, value in self.sol_options.items():
                    if self.sol_options[key]['Slack'] is True:
                        self.vis_gui.slack_dict[key].set_ydata(np.array(self.sol_options[key]['Slack_Sol']))

                # self.vis_gui.main_fig.canvas.draw()

                if self.vis_options['b_save_tikz']:
                    self.vis_gui.main_fig.canvas.draw()

                    tikzplotlib.save('SQP_OSQP_debug.tex')

            if not self.b_vis_triggered and self.vis_options['b_movie']:

                # --- Reset trigger once movie is running
                self.b_vis_triggered = True

                # --- Allow interactive mode
                plt.ion()

                for i in range(0, self.row_count - self.log_lines, self.log_lines):
                    self.vis_gui.slider_vel.set_val(int(i))
                    print("\n*** Progress: " + str("%.2f" % round((i / self.row_count) * 100, 2)) + " %")

                # --- Plot runtimes
                for key, value in self.sol_options.items():

                    if self.vis_options['b_calc_time_plot']:
                        self.runtimes.plot_hist(name_solver=self.sol_options[key]['Solver'],
                                                dt_solver=np.array(self.sol_options[key]['Time']),
                                                vis_options=self.vis_options)

                plt.ioff()

            # --- Show all plots that have been calculated
            plt.show()

    def read_log_file(self,
                      csv_name: str,
                      idx: int) -> list:

        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.02.2020

        Documentation: Reads specific line in log file and splits into its variables if several exist in this line.

        Inputs:
        csv_name: file name and path to read data from
        idx: the starting line-number of the data block that should be read from the log-file

        Outputs:
        list containing all logged variable values
        """

        row_lc = linecache.getline(csv_name, idx + 1)

        # --- In case of RAM trouble but calculation speed will drop significantly
        # linecache.clearcache()

        # --- Get rid of ending '\n'
        return row_lc[:-1].rsplit(';')


if __name__ == '__main__':
    pass
