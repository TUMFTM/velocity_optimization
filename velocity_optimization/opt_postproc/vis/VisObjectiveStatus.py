from matplotlib import pyplot as plt
from velocity_optimization.opt_postproc.src.CalcObjective import CalcObjective
try:
    import tikzplotlib
except ImportError:
    print('Warning: No module tikzplotlib found. Not necessary on car but for development.')

# Line Width
LW = 1.5


class VisVP_ObjStatus:

    __slots__ = ('calc_objective',
                 'obj_status_fig')

    def __init__(self,
                 calc_objective: CalcObjective,
                 vis_options: dict):
        """
        Python version: 3.5
        Created by: Thomas Herrmann (thomas.herrmann@tum.de)
        Created on: 01.02.2020

        Documentation: Class to visualize the objective function terms and the QP status for the logged race session.

        Inputs:
        calc_objective: object containing the object function values
        vis_options: user specified options for the debug visualization
        """

        self.calc_objective = calc_objective

        self.obj_status_fig = plt.figure()

        ########################################################################################################
        # --- QP info
        ########################################################################################################
        ax1 = self.obj_status_fig.add_subplot(4, 1, 1)
        # Rm x tick labels
        ax1.set_xticklabels([])

        p1, = ax1.plot(self.calc_objective.qp_iter_last_arr,
                       color='black', linewidth=LW, linestyle='-')

        plt.ylabel(r'$\mathrm{QP}_\mathrm{iter}$')

        ax2 = self.obj_status_fig.add_subplot(4, 1, 2)
        ax2.set_xticklabels([])

        p1, = ax2.plot(self.calc_objective.qp_status_last_arr,
                       color='black', linewidth=LW, linestyle='-')

        plt.ylabel(r'$\mathrm{QP}_\mathrm{stat}$')
        plt.xlabel(r'$v_i$')

        ########################################################################################################
        # --- Cost function
        ########################################################################################################
        ax3 = self.obj_status_fig.add_subplot(3, 1, 3)
        ax3.get_shared_x_axes().join(ax3, ax1)
        ax3.get_shared_x_axes().join(ax3, ax2)

        p1, = ax3.plot(self.calc_objective.J_vvmax,
                       color='black', linewidth=LW, linestyle='-')
        p1, = ax3.plot(self.calc_objective.J_eps_tre_lin,
                       color='blue', linewidth=LW, linestyle='--')
        p1, = ax3.plot(self.calc_objective.J_eps_tre_quad,
                       color='green', linewidth=LW, linestyle=':')
        p1, = ax3.plot(self.calc_objective.J_jerk,
                       color='red', linewidth=LW, linestyle='-.')

        plt.ylabel(r'$J$')
        plt.xlabel(r'$v_i$')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        plt.legend([r'$J_\mathrm{v_\mathrm{max}}$',
                    r'$J_\mathrm{\epsilon,lin}$',
                    r'$J_\mathrm{\epsilon,quad}$',
                    r'$J_\mathrm{\dot{a}}$'])

        if vis_options['b_save_tikz']:
            tikzplotlib.save("SQP_OSQP_objective.tex")

        # --- Show main window
        # plt.show()


if __name__ == '__main__':
    pass
