import numpy as np
import datetime
import time
from matplotlib import pyplot as plt
import matplotlib.patches as pltpatches
from matplotlib.widgets import Slider, Button
from velocity_optimization.src.params_vp_sqp import params_vp_sqp
from velocity_optimization.opt_postproc.src.CalcObjective import CalcObjective

# Font sizes
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

# Line Width
LW = 1.5

# TUM Color
TUMBlue = [0 / 255, 101 / 255, 189 / 255]
TUMWhite = [255 / 255, 255 / 255, 255 / 255]
TUMBlack = [0 / 255, 0 / 255, 0 / 255]
TUMBlue1 = [0 / 255, 51 / 255, 89 / 255]
TUMBlue2 = [0 / 255, 82 / 255, 147 / 255]
TUMGray1 = [51 / 255, 51 / 255, 51 / 255]
TUMGray2 = [127 / 255, 127 / 255, 127 / 255]
TUMGray3 = [204 / 255, 204 / 255, 204 / 255]
TUMBlue3 = [100 / 255, 160 / 255, 200 / 255]
TUMBlue4 = [152 / 255, 198 / 255, 234 / 255]
TUMIvory = [218 / 255, 215 / 255, 203 / 255]
TUMOrange = [227 / 255, 114 / 255, 34 / 255]
TUMGreen = [162 / 255, 173 / 255, 0 / 255]

# controls default text sizes
plt.rc('font', size=SMALL_SIZE)
# fontsize of the axes title
plt.rc('axes', titlesize=SMALL_SIZE)
# fontsize of the x and y labels
plt.rc('axes', labelsize=MEDIUM_SIZE)
# fontsize of the tick labels
plt.rc('xtick', labelsize=SMALL_SIZE)
# fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)
# legend fontsize
plt.rc('legend', fontsize=SMALL_SIZE)
# fontsize of the figure title
plt.rc('figure', titlesize=BIGGER_SIZE)

Y_V_MAX_MPS = 70

Y_FORCE_MIN_KN = -20
Y_FORCE_MAX_KN = 8

Y_DELTAFORCE_MIN_KN = -5.5
Y_DELTAFORCE_MAX_KN = 5.5

Y_POWER_MIN_KW = -600
Y_POWER_MAX_KW = 280

Y_SLACK_TRE_MIN = -0.5
Y_SLACK_TRE_MAX = 5.0

X_AYMAX_MPS2 = 13
Y_AXMAX_MPS2 = 13


class VisVP_Logs_GUI:

    __slots__ = ('m',
                 'vis_handler',
                 'vis_options',
                 'sol_options',
                 'main_fig',
                 'params_opt',
                 'constants_opt',
                 'n',
                 'text_status',
                 'slider_vel',
                 'but_next',
                 'but_prev',
                 'vel_dict', 'F_dict', 'P_dict', 'a_dict', 'ax_dict', 'ay_dict', 'F_f_dict', 'F_r_dict', 'F_fl_dict',
                 'F_fr_dict', 'F_rl_dict', 'F_rr_dict', 'slack_dict',
                 'p1_1', 'p1_2', 'p1_3', 'p1_4', 'p1_5', 'p1_6', 'p1_7', 'p1_8', 'p1_9', 'p1_10', 'p1_11', 'p1_11',
                 'p1_12', 'p1_13', 'p1_14', 'p1_15', 'p1_16', 'p1_17', 'p1_18',
                 'p3_1', 'p3_2', 'p3_3', 'p3_4', 'p3_5', 'p3_6', 'p3_7', 'p3_8', 'p3_9', 'p3_10', 'p3_11', 'p3_12',
                 'p3_13', 'p3_14', 'p3_15',
                 'p4_1',
                 'p5_1', 'p5_2', 'p5_3', 'p5_4', 'p5_5', 'p5_6', 'p5_7', 'p5_8', 'p5_9', 'p5_10', 'p5_11', 'p5_12',
                 'p5_13', 'p5_14', 'p5_15',
                 'p6_1', 'p6_2', 'p6_3', 'p6_4', 'p6_5', 'p6_6', 'p6_7', 'p6_8', 'p6_9', 'p6_10', 'p6_11', 'p6_12',
                 'p6_13', 'p6_14', 'p6_15', 'p6_16',
                 'p7_1', 'p7_2', 'p7_3', 'p7_4', 'p7_5', 'p7_6')

    def __init__(self,
                 vis_handler,
                 m: int,
                 vis_options: dict,
                 params_path: str,
                 sol_options: dict):

        """
        Python version: 3.5
        Created by: Thomas Herrmann (thomas.herrmann@tum.de)
        Created on: 01.02.2020
        Modified by: Tobias Klotz

        Documentation: Class creating a GUI to visualize logged data from the velocity optimization SQP.

        Inputs:
        vis_handler: main visualization handler object of type 'VisVP_Logs'
        m: number of velocity optimization variables
        vis_options: user specified visualization options
        params_path: absolute path to folder containing config file .ini
        sol_options: user specified solver options
        """

        self.m = m
        self.vis_handler = vis_handler
        self.vis_options = vis_options
        self.sol_options = sol_options
        self.main_fig = None

        self.params_opt = params_vp_sqp(m=m,
                                        sid=vis_handler.sid,
                                        params_path=params_path)[0]

        self.constants_opt = CalcObjective(csv_name='empty',
                                           log_lines=0,
                                           sid=vis_handler.sid,
                                           params_path=params_path)

        # number of slack variables
        self.n = np.int(np.ceil(m / self.params_opt['slack_every_v']))

        # Text for QP status within logged data
        self.text_status = None

        # Slider to select logged data
        self.slider_vel = None

        # Command buttons to load logged data
        self.but_next = None
        self.but_prev = None

        # Initialize solution dictionaries
        self.vel_dict = {}
        self.slack_dict = {}
        self.F_dict = {}
        self.P_dict = {}
        self.a_dict = {}
        self.ax_dict = {}
        self.ay_dict = {}
        self.F_f_dict = {}
        self.F_r_dict = {}
        self.F_fl_dict = {}
        self.F_fr_dict = {}
        self.F_rl_dict = {}
        self.F_rr_dict = {}

        # Lines within debug plots
        self.p1_1, self.p1_2, self.p1_3, self.p1_4, self.p1_5, self.p1_6, self.p1_7, self.p1_8, self.p1_9, self.p1_10, \
            self.p1_11, self.p1_12, self.p1_13, self.p1_14, self.p1_15, self.p1_16, self.p1_17, self.p1_18 = \
            None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        self.p3_1, self.p3_2, self.p3_3, self.p3_4, self.p3_5, self.p3_6, self.p3_7, self.p3_8, self.p3_9, \
            self.p3_10, self.p3_11, self.p3_12, self.p3_13, self.p3_14, self.p3_15 = \
            None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        self.p4_1 = None
        self.p5_1, self.p5_2, self.p5_3, self.p5_4, self.p5_5, self.p5_6, self.p5_7, self.p5_8, self.p5_9, self.p5_10, \
            self.p5_11, self.p5_12, self.p5_13, self.p5_14, self.p5_15 = \
            None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        self.p6_1, self.p6_2, self.p6_3, self.p6_4, self.p6_5, self.p6_6, self.p6_7, self.p6_8, self.p6_9, self.p6_10, \
            self.p6_11, self.p6_12, self.p6_13, self.p6_14, self.p6_15, self.p6_16 = \
            None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        self.p7_1, self.p7_2, self.p7_3, self.p7_4, self.p7_5, self.p7_6 = None, None, None, None, None, None

        self.line_definition()
        self.draw_gui()
        self.initialize_plots()

    def line_definition(self):
        # Define Color, Line Style and Marker
        for key, value in self.sol_options.items():
            if self.sol_options[key]['Model'] == "PM":
                self.sol_options[key].update({'Color': TUMBlue, 'Linestyle': ':', 'Marker': 'o'})
            if self.sol_options[key]['Model'] == "KM":
                self.sol_options[key].update({'Color': TUMOrange, 'Linestyle': '--', 'Marker': 'x'})
            if self.sol_options[key]['Model'] == "DM":
                self.sol_options[key].update({'Color': TUMGreen, 'Linestyle': '-.', 'Marker': 'v'})
            if self.sol_options[key]['Model'] == "FW":
                self.sol_options[key].update({'Color': TUMGray1, 'Linestyle': '-', 'Marker': 's'})

    def draw_gui(self):

        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.02.2020

        Documentation: Creates a main windows including interactive objects in GUI to visualize logged data.
        """

        # --- Draw main window for logs visualization
        self.main_fig = plt.figure()

        self.main_fig.add_subplot(100, 2, 1)
        self.main_fig.subplots_adjust(bottom=0.1)

        self.text_status = plt.text(0, 5,
                                    r'$s_\mathrm{glob}$: ' + str('%.2f' % 0) + ' m' + r'   $t$: '
                                    + str(3600 + time.mktime(
                                        datetime.datetime.strptime(
                                            '2020-01-01-01:01:01.00001', "%Y-%m-%d-%H:%M:%S.%f").timetuple())) + ' s'
                                    + r'   $\mathrm{QP}_\mathrm{status}$: ' + str(0)
                                    + r'   $\mathrm{QP}_\mathrm{iter}$: ' + str(0)
                                    + r'   $\mathrm{SQP}_\mathrm{\Delta t}$: ' + str(0) + ' ms')
        plt.axis('off')

        ################################################################################################################
        # --- Slider and Button update functions
        ################################################################################################################
        def update(val):
            self.slider_vel.valtext.set_text(int(val / self.vis_handler.log_lines))
            self.main_fig.canvas.draw_idle()
            self.vis_handler.vis_log(int(val))

        def next_val(event):
            self.slider_vel.set_val(self.slider_vel.val + self.vis_handler.log_lines)
            plt.draw()

        def prev_val(event):
            self.slider_vel.set_val(self.slider_vel.val - self.vis_handler.log_lines)
            plt.draw()

        ax_freq = plt.axes([0.1, 0.01, 0.8, 0.02])
        self.slider_vel = Slider(ax_freq, 'Id', 0, self.vis_handler.row_count,
                                 valinit=0)
        # self.slider_vel = Slider(ax_freq, 'Id', 0, self.vis_handler.row_count,
        # valinit=0, valstep=self.vis_handler.log_lines)

        self.slider_vel.on_changed(update)

        ################################################################################################################
        # --- Prev/Next Buttons
        ################################################################################################################
        ax_next = plt.axes([0.95, 0.0, 0.04, 0.04])
        self.but_next = Button(ax_next, 'Next')
        self.but_next.on_clicked(next_val)

        ax_prev = plt.axes([0.95, 0.05, 0.04, 0.04])
        self.but_prev = Button(ax_prev, 'Prev')
        self.but_prev.on_clicked(prev_val)

    def initialize_plots(self):

        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.02.2020

        Documentation: Initializes the plots within the debug GUI.
        """

        x_dots = np.linspace(0, self.m - 2, self.m - 1)
        x_dots_red = np.linspace(0, self.m - 2 - 1, self.m - 1 - 1)

        ################################################################################################################
        # --- Velocity
        ################################################################################################################
        ax1 = self.main_fig.add_subplot(4, 2, 1)
        ax = ax1

        ax.set_ylim([-1, Y_V_MAX_MPS])

        p1, = ax.plot(np.zeros((self.m, 1)),
                      color='black', linewidth=LW, linestyle='-')
        p2, = ax.plot(np.zeros((self.m, 1)),
                      color='red', linewidth=LW, linestyle='--')
        p3, = ax.plot(self.m - 1, 0, marker='x',
                      color='green', linewidth=LW, markersize=LW * 3)
        p4, = ax.plot(np.zeros((self.m, 1)),
                      color='gray', linewidth=LW, linestyle=':')
        legend = [r'$v_\mathrm{o,OSQP}$',
                  r'$v_\mathrm{max}$',
                  r'$v_\mathrm{end}$',
                  r'$v_\mathrm{ini}$']

        for key, value in self.sol_options.items():
            self.vel_dict[key], = ax.plot(np.zeros((self.m, 1)), color=self.sol_options[key]["Color"], linewidth=LW,
                                          linestyle=self.sol_options[key]["Linestyle"],
                                          marker=self.sol_options[key]["Marker"], markevery=5)
            legend.append(r'$v_\mathrm{o,{%s,%s}}$' % (self.sol_options[key]["Solver"],
                                                       self.sol_options[key]["Model"]))

        self.p1_1 = p1
        self.p1_2 = p2
        self.p1_3 = p3
        self.p1_4 = p4

        plt.ylabel(r'$v$' + ' in ' r'$\mathrm{\frac{m}{s}}$')

        plt.legend(legend,
                   mode='expand', ncol=11)

        ################################################################################################################
        # --- Force
        ################################################################################################################
        ax2 = self.main_fig.add_subplot(4, 2, 3)
        ax = ax2

        ax.set_ylim([Y_FORCE_MIN_KN, Y_FORCE_MAX_KN])
        # --- Rm x tick labels
        ax.set_xticklabels([])

        legend = [r'$F_\mathrm{o,OSQP}$',
                  r'tol']
        p1, = ax.plot(x_dots,
                      np.zeros((self.m - 1, 1)),
                      color='black', linewidth=LW, linestyle='-')
        p2, = ax.plot([0, 0, 0],
                      [0,
                       0 + 1,
                       0 - 1],
                      linestyle='-', marker='_', color='red', linewidth=LW, markersize=LW * 3)

        for key, value in self.sol_options.items():
            self.F_dict[key], = ax.plot(np.zeros((self.m - 1, 1)), color=self.sol_options[key]["Color"], linewidth=LW,
                                        linestyle=self.sol_options[key]["Linestyle"],
                                        marker=self.sol_options[key]["Marker"], markevery=5)
            legend.append(r'$F_\mathrm{o,%s,%s}$' % (self.sol_options[key]["Solver"],
                                                     self.sol_options[key]["Model"]))

        self.p3_1 = p1
        self.p3_2 = p2

        plt.ylabel(r'$F$' + ' in ' r'$\mathrm{kN}$')
        plt.legend(legend,
                   mode='expand', ncol=5)

        ################################################################################################################
        # --- Delta Force
        ################################################################################################################
        ax3 = self.main_fig.add_subplot(4, 2, 5)
        ax = ax3

        ax.set_ylim([Y_DELTAFORCE_MIN_KN, Y_DELTAFORCE_MAX_KN])
        ax.set_xticklabels([])

        p1, = ax.plot(x_dots_red,
                      np.zeros((self.m - 2, 1)),
                      color='black', linewidth=LW, linestyle='-')

        self.p4_1 = p1

        plt.ylabel(r'$\Delta F$' + ' in ' r'$\mathrm{kN}$')

        ################################################################################################################
        # --- Power
        ################################################################################################################
        ax4 = self.main_fig.add_subplot(4, 2, 7)
        ax = ax4

        ax.set_ylim([Y_POWER_MIN_KW, Y_POWER_MAX_KW])
        legend = [r'$P$',
                  r'$P_\mathrm{max}$']
        if self.params_opt['b_var_power']:
            p1, = ax.plot(x_dots,
                          np.zeros((self.m - 1, 1)),
                          color='black', linewidth=LW, linestyle='-')
        else:
            p1, = ax.plot(x_dots,
                          np.zeros((self.m - 1, 1)),
                          color='black', linewidth=LW, linestyle='-')

        # P_max_kW-limit
        p2, = ax.plot(np.zeros((self.m - 1, 1)),
                      color='red', linewidth=LW, linestyle='--')

        for key, value in self.sol_options.items():
            self.P_dict[key], = ax.plot(np.zeros((self.m - 1, 1)), color=self.sol_options[key]["Color"], linewidth=LW,
                                        linestyle=self.sol_options[key]["Linestyle"],
                                        marker=self.sol_options[key]["Marker"], markevery=5)
            legend.append(r'$P_\mathrm{o,%s,%s}$' % (self.sol_options[key]["Solver"],
                                                     self.sol_options[key]["Model"]))

        self.p5_1 = p1
        self.p5_2 = p2
        self.p5_3 = p3

        plt.ylabel(r'$P$' + ' in ' r'$\mathrm{kW}$')
        plt.xlabel(r'$s_{\mathrm{loc},j}$')

        plt.legend(legend,
                   loc='lower right', ncol=2)

        # --- Combine x-axes
        ax4.get_shared_x_axes().join(ax4, ax1)
        ax4.get_shared_x_axes().join(ax4, ax2)
        ax4.get_shared_x_axes().join(ax4, ax3)

        ################################################################################################################
        # --- Slack variables
        ################################################################################################################
        ax = self.main_fig.add_subplot(5, 2, 2)
        ax.set_ylim([Y_SLACK_TRE_MIN, Y_SLACK_TRE_MAX])
        legend = [r'$\epsilon_\mathrm{o,OSQP}$',
                  r'$\epsilon_\mathrm{ini}$',
                  r'$\epsilon_\mathrm{max}$']

        p1, = ax.plot(np.zeros((self.n, 1)),
                      color='black', linewidth=LW, linestyle='-')
        p2, = ax.plot(np.zeros((self.n, 1)),
                      color='gray', linewidth=LW, linestyle=':')
        p3, = ax.plot(np.zeros((self.n, 1)),
                      color='red', linewidth=LW, linestyle='--')

        for key, value in self.sol_options.items():
            if self.sol_options[key]['Slack'] is True:
                self.slack_dict[key], = ax.plot(np.zeros((self.n, 1)), color=self.sol_options[key]["Color"],
                                                linewidth=LW,
                                                linestyle=self.sol_options[key]["Linestyle"])
                legend.append(r'$\epsilon_\mathrm{o,%s,%s}$' % (self.sol_options[key]["Solver"],
                                                                self.sol_options[key]["Model"]))

        self.p7_1 = p1
        self.p7_2 = p2
        self.p7_3 = p3

        plt.xlabel(r'$s_{\mathrm{loc},k}$')
        plt.ylabel(r'$\epsilon$')

        plt.legend(legend,
                   mode='expand', ncol=5)

        ########################################################################################################
        # --- Tire usage
        ########################################################################################################
        ax = self.main_fig.add_subplot(2, 2, 4)
        ax.set_xlim([- X_AYMAX_MPS2, X_AYMAX_MPS2])
        ax.set_ylim([- Y_AXMAX_MPS2, Y_AXMAX_MPS2])

        legend = [r'$\bar{\mu}$',
                  r'$\bar{\mu} + \epsilon_\mathrm{max}$',
                  r'$\mu_\mathrm{o}$']

        # --- Case of constant friction limits
        p1, = ax.plot([self.constants_opt.params['aymax_mps2_'],
                       0,
                       - self.constants_opt.params['aymax_mps2_'],
                       0,
                       self.constants_opt.params['aymax_mps2_']],
                      [0,
                       self.constants_opt.params['axmax_mps2_'],
                       0,
                       - self.constants_opt.params['axmax_mps2_'],
                       0],
                      color='green', linestyle='-', linewidth=LW)

        p2, = ax.plot([self.constants_opt.params['axmax_mps2_']
                       * (1 + self.constants_opt.params['s_v_t_lim_'] * self.constants_opt.params['s_v_t_unit_']),
                       0,
                       - self.constants_opt.params['axmax_mps2_']
                       * (1 + self.constants_opt.params['s_v_t_lim_'] * self.constants_opt.params['s_v_t_unit_']),
                       0,
                       self.constants_opt.params['axmax_mps2_']
                       * (1 + self.constants_opt.params['s_v_t_lim_'] * self.constants_opt.params['s_v_t_unit_'])],
                      [0,
                       self.constants_opt.params['axmax_mps2_']
                       * (1 + self.constants_opt.params['s_v_t_lim_'] * self.constants_opt.params['s_v_t_unit_']),
                       0,
                       - self.constants_opt.params['axmax_mps2_']
                       * (1 + self.constants_opt.params['s_v_t_lim_'] * self.constants_opt.params['s_v_t_unit_']),
                       0],
                      color='green', linestyle='--', linewidth=LW)

        # Plot ellipse
        for key, value in self.sol_options.items():
            if self.sol_options[key]['Friction'] == 'Circle':
                ax.add_patch(pltpatches.Ellipse((0, 0), self.constants_opt.params['aymax_mps2_'] * 2,
                                                self.constants_opt.params['axmax_mps2_'] * 2,
                                                facecolor='none', edgecolor='green', linewidth=LW, alpha=0.5))

        p, = ax.plot(np.zeros((self.m - 1, 1)), np.zeros((self.m - 1, 1)),
                     marker='x', linestyle='', markersize=LW * 3, color='black')

        for key, value in self.sol_options.items():
            self.a_dict[key], = ax.plot(np.zeros((self.m - 1, 1)), np.zeros((self.m - 1, 1)),
                                        color=self.sol_options[key]["Color"], linewidth=LW,
                                        linestyle=self.sol_options[key]["Linestyle"],
                                        marker=self.sol_options[key]["Marker"], markevery=2)
            legend.append(r'$\mu_\mathrm{o,%s,%s}$' % (self.sol_options[key]["Solver"],
                                                       self.sol_options[key]["Model"]))

        ax.axis('equal')

        # --- Case of variable friction limits draw maximum available friction limits
        if self.params_opt['b_var_friction']:
            axmax_max_plot = 0
            axmax_min_plot = 0
            aymax_max_plot = 0
            aymax_min_plot = 0

            p3, = ax.plot([aymax_max_plot, 0, - aymax_max_plot, 0, aymax_max_plot],
                          [0, axmax_max_plot, 0, - axmax_max_plot, 0],
                          color='gray', linestyle='-', linewidth=LW)

            p4, = ax.plot([aymax_min_plot, 0, - aymax_min_plot, 0, aymax_min_plot],
                          [0, axmax_min_plot, 0, - axmax_min_plot, 0],
                          color='gray', linestyle='--', linewidth=LW)

        self.p6_1 = p
        self.p6_2 = p3
        self.p6_3 = p4

        plt.xlabel(r'$a_\mathrm{y}$' + ' in ' + r'$\frac{m}{s^2}$')
        plt.ylabel(r'$a_\mathrm{x}$' + ' in ' + r'$\frac{m}{s^2}$')

        plt.legend(legend)

        # --- Draw maximum motor acceleration potential
        ax.plot([- self.constants_opt.params['axmax_mps2_'], self.constants_opt.params['axmax_mps2_']],
                [self.constants_opt.params['Fmax_kN_'] / self.constants_opt.params['m_t_'],
                 self.constants_opt.params['Fmax_kN_'] / self.constants_opt.params['m_t_']],
                linestyle='--', color='red', linewidth=LW)

        for key, value in self.sol_options.items():

            if self.sol_options[key]['Model'] == "DM" or self.sol_options[key]['Model'] == "FW":

                ########################################################################################################
                # --- Front Axle Tire usage
                ########################################################################################################
                ax = self.main_fig.add_subplot(4, 5, 9)
                ax.set_xlim([- 1, 1])
                ax.set_ylim([- 1, 1])

                circle = plt.Circle((0, 0), 1, facecolor='none',
                                    edgecolor=TUMGray1, linewidth=LW, alpha=0.5)
                ax.add_patch(circle)

                for key, value in self.sol_options.items():
                    if self.sol_options[key]['Model'] == "DM":
                        self.F_f_dict[key], = ax.plot(np.zeros((self.m - 1, 1)), np.zeros((self.m - 1, 1)),
                                                      color=self.sol_options[key]["Color"], linewidth=LW,
                                                      linestyle=self.sol_options[key]["Linestyle"],
                                                      marker=self.sol_options[key]["Marker"], markevery=2)
                        legend.append(r'$F_\mathrm{o,%s,%s}$' % (self.sol_options[key]["Model"],
                                                                 self.sol_options[key]["Solver"]))

                    elif self.sol_options[key]['Model'] == "FW" and self.sol_options[key]['Solver'] == "IPOPT":
                        self.F_fl_dict[key], = ax.plot(np.zeros((self.m - 1, 1)), np.zeros((self.m - 1, 1)),
                                                       color=self.sol_options[key]["Color"], linewidth=LW,
                                                       linestyle=self.sol_options[key]["Linestyle"],
                                                       marker=self.sol_options[key]["Marker"], markevery=2)
                        self.F_fr_dict[key], = ax.plot(np.zeros((self.m - 1, 1)), np.zeros((self.m - 1, 1)),
                                                       color=TUMGreen, linewidth=LW,
                                                       linestyle=self.sol_options[key]["Linestyle"],
                                                       marker='*', markevery=2)

                plt.xlabel(r'$F_\mathrm{y,f} / F_\mathrm{z,f}$')
                plt.ylabel(r'$F_\mathrm{x,f} / F_\mathrm{z,f}$')
                plt.title('Front Tire')
                if self.sol_options[key]['Model'] == "FW":
                    plt.legend(['left', 'right'])

                ax.axis('equal')

                ########################################################################################################
                # --- Rear Axle Tire usage
                ########################################################################################################
                ax = self.main_fig.add_subplot(4, 5, 10)
                ax.set_xlim([- 1, 1])
                ax.set_ylim([- 1, 1])

                circle = plt.Circle((0, 0), 1, facecolor='none',
                                    edgecolor=TUMGray1, linewidth=LW, alpha=0.5)
                ax.add_patch(circle)

                for key, value in self.sol_options.items():
                    if self.sol_options[key]['Model'] == "DM":
                        self.F_r_dict[key], = ax.plot(np.zeros((self.m - 1, 1)), np.zeros((self.m - 1, 1)),
                                                      color=self.sol_options[key]["Color"], linewidth=LW,
                                                      linestyle=self.sol_options[key]["Linestyle"],
                                                      marker=self.sol_options[key]["Marker"], markevery=2)
                    elif self.sol_options[key]['Model'] == "FW" and self.sol_options[key]['Solver'] == "IPOPT":
                        self.F_rl_dict[key], = ax.plot(np.zeros((self.m - 1, 1)), np.zeros((self.m - 1, 1)),
                                                       color=self.sol_options[key]["Color"], linewidth=LW,
                                                       linestyle=self.sol_options[key]["Linestyle"],
                                                       marker=self.sol_options[key]["Marker"], markevery=2)
                        self.F_rr_dict[key], = ax.plot(np.zeros((self.m - 1, 1)), np.zeros((self.m - 1, 1)),
                                                       color=TUMGreen, linewidth=LW,
                                                       linestyle=self.sol_options[key]["Linestyle"],
                                                       marker='*', markevery=2)

                plt.xlabel(r'$F_\mathrm{y,r} / F_\mathrm{z,r}$')
                plt.title('Rear Tire')
                if self.sol_options[key]['Model'] == "FW":
                    plt.legend(['left', 'right'])

                ax.axis('equal')
                # --- Show main window
                # plt.show()


if __name__ == '__main__':
    pass
