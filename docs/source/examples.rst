********
Examples
********

This section describes how to use the velocity_optimization package.

Velocity optimization
=====================

The following class `VelOpt` configures a velocity optimizer object `v_sqp`. `VelOpt` also contains a function
`vsqp_online`that needs the input of e.g., the road curvature `kappa` and the step length between the
discretization points `delta_s`. Additionally, the current vehicle velocity `v_ini`, the initial velocity guess `x0_v`
and the force `F_ini` from the first velocity point (to avoid oscillations in subsequent SQPs) should be provided.


.. code-block:: python

    import os
    import sys
    import numpy as np
    import velocity_optimization


    class VelOpt():

        def __init__(self,
                     m: int):
            self.vsqp_setup(m=m)

        def vsqp_setup(self,
                       m: int):
            self.v_sqp = velocity_optimization.src.VelQP. \
                VelQP(m=m,
                      sid='PerfSQP',
                      params_path=os.path.dirname(os.path.abspath(__file__)) + '/params/',
                      input_path=os.path.dirname(os.path.abspath(__file__)) + '/inputs/',
                      logging_path=os.path.dirname(os.path.abspath(__file__)) + '/logs/')

        def vsqp_online(self,
                        v_ini: float,
                        kappa: np.ndarray,
                        delta_s: np.ndarray,
                        x0_v: np.ndarray,
                        F_ini: float):

            v_op, s_t_op, qp_status = velocity_optimization.src.online_qp. \
                online_qp(velqp=self.v_sqp,
                          v_ini=v_ini,
                          kappa=kappa,
                          delta_s=delta_s,
                          x0_v=x0_v,
                          v_max=np.array([70] * self.v_sqp.m),
                          v_end=5,
                          F_ini=F_ini,
                          s_glob=0,
                          ax_max=10 * np.ones((self.v_sqp.m, )),
                          ay_max=10 * np.ones((self.v_sqp.m, )))

            return v_op, s_t_op, qp_status


    if __name__ == '__main__':
        m = 115
        v = VelOpt(m=m)

        v_op, _, qp_status = v.vsqp_online(v_ini=10,
                                           kappa=0.0001 * np.ones((m,)),
                                           delta_s=2 * np.ones((m - 1,)),
                                           x0_v=25 * np.ones((m,)),
                                           F_ini=3)

        print("Optimized velocity profile:\n", v_op)


Debugging
=========

.. image:: GUI_Plot.png
   :width: 700

Our package comes with a powerful debugging tool. Create an empty file and copy the following content to this file.
Adapt `csv_name`, `params_path` and `input_path` to your specific paths. A debug window will show up, plotting the
most important values of the velocity SQP that have been logged.

.. code-block:: python

    import os
    import sys
    import numpy as np
    import linecache
    import json
    from matplotlib import pyplot as plt

    # custom modules
    vel_opt_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(vel_opt_path)
    from velocity_optimization.opt_postproc.vis.VisBenchmarkLogs import VisVP_Logs


    if __name__ == "__main__":
        """
        Python version: 3.5
        Created by: Thomas Herrmann (thomas.herrmann@tum.de)
        Created on: 01.02.2020
        Modified by: Tobias Klotz

        Documentation: This function visualizes calculated velocity from the SQP-planner including its constraints.
        """

        # ------------------------------------------------------------------------------------------------------------------
        # USER INPUT -------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------
        csv_name = vel_opt_path + '/logs/sqp_perf_2020_12_26_10_33.log'

        # TODO: check the paths to your 'params'- and 'inputs'-folders here and adapt if necessary!
        params_path = vel_opt_path + '/params/'
        input_path = vel_opt_path + '/inputs/'

        b_movie = False                      # visualize all logs consecutively?

        b_run_OSQP = False                   # re-calculate QP from log-input?

        b_idx = 0                            # Choose Starting Idx of Log-File

        b_global_plot = False                # do global plot of states for entire log?
        glob_lim = np.inf

        b_immediate_plot_update = True       # plot immediately or only solver data replay?

        b_calc_time_plot = True              # show plot of solver runtimes?

        b_save_tikz = False                  # save plots as tikz files?

        # --- Define solver options for IPOPT as benchmark solution
        sol_options = {'solver1': {'Model': "FW",               # PM (Pointmass), KM (Kinematic Single Track Model),
                                                                # DM (Dynamic Single Track Model), FW (Double Track Model)
                                   'Solver': "IPOPT",           # IPOPT, qpOASES
                                   'Friction': "Diamond",       # Circle, Diamond (only for PM and KM, rest have Circles)
                                   'VarFriction': True,         # Variable friction: True, False
                                   'VarPower': False,           # Variable power: True, False
                                   'Slack': True,               # Usage of slack variables on comb. acceleration (only
                                                                # reasonable on simple models like PM and KM): Keep True!
                                   }
                       }

        # ------------------------------------------------------------------------------------------------------------------
        # END USER INPUT ---------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------

        csv_name_ltpl = None

        for key, value in sol_options.items():
            if sol_options[key]['Slack'] and (sol_options[key]['Model'] == 'DM' or sol_options[key]['Model'] == 'FW'):
                sol_options[key]['Slack'] = False

        # Number of log lines spanning one data block
        log_lines = 4

        # --- Transform ID of used velocity planner into 'PerfSQP' or 'EmergSQP'
        sid = csv_name.split('/')[-1].split('_')[1]
        if sid == 'perf':
            sid = 'PerfSQP'
        elif sid == 'emerg':
            sid = 'EmergSQP'
        else:
            print('Logs have been produced with illegal SID! Exiting.')
            sys.exit(1)

        # --- Number of velocity points
        # Get length of velocity array to determine parameter 'm' in velocity optimization
        row_lc = linecache.getline(csv_name, 1)
        row_lc = row_lc[:-1].rsplit(';')
        velocity_dummy = json.loads(row_lc[2])
        m = len(velocity_dummy)

        # visualization options
        vis_options = {'b_movie': b_movie,
                       'b_run_OSQP': b_run_OSQP,
                       'b_idx': b_idx,
                       'b_global_plot': b_global_plot,
                       'glob_lim': glob_lim,
                       'b_immediate_plot_update': b_immediate_plot_update,
                       'b_calc_time_plot': b_calc_time_plot,
                       'b_save_tikz': b_save_tikz}

        # --- Set up visualization object
        rL = VisVP_Logs(csv_name=csv_name,
                        csv_name_ltpl=csv_name_ltpl,
                        m=m,
                        sid=sid,
                        log_lines=log_lines,
                        vis_options=vis_options,
                        params_path=params_path,
                        input_path=input_path,
                        sol_options=sol_options)

        # --- Start GUI
        rL.vis_log(int(0))

        # --- Show main window
        plt.show()

General options
---------------

There are several user options that can be changed for the visualization:

.. list-table:: Visualization Options
   :widths: 25 10 65
   :header-rows: 1

   * - Name
     - Value
     - Description
   * - csv_name
     - Path
     - Path to the log-file.
   * - params_path
     - Path
     - Path to the directory containing the velocity planner configuration file.
   * - input_path
     - Path
     - Path to the directory of the input data (variable power/friction data).
   * - log_lines
     - Int
     - | Number of lines in the log-file which belong to a single planning horizon. See more information at the
       | description of the log-file structure.
   * - b_movie
     - True/False
     - Choose if all entries in log file shall be run subsequently without stopping between different planning horizons.
   * - b_run_OSQP
     - True/False
     - Choose if the optimization problem is re-solved with the OSQP solver (reference solver).
   * - b_calc_qpOASES
     - True/False
     - Choose if the optimization problem is solved with the solver qpOASES.
   * - b_idx
     - Int
     - Select a specific planning horizon to be plotted in the GUI. Choose 0 to disable this feature.
   * - b_global_plot
     - True/False
     - Debug energy values for entire run.
   * - glob_lim
     - True/False
     - Set a limit for the last ID of logs that should be included in the global plots.
   * - b_immediate_plot_update
     - True/False
     - Update the plots in the GUI after solving the optimization problem for each planning horizon.
   * - b_calc_time_plot
     - True/False
     - Show and update solver runtime histogram.
   * - b_save_tikz
     - True/False
     - Save the solver runtime histograms.

Available solver and model combinations
---------------------------------------

Additionally, the following combinations of debugging solvers and vehicle models are available to compare the solution
that
was
calculated using OSQP during driving. The word in the cells below indicate the available combined acceleration models:

+------------+------------+-----------+
|            | IPOPT      | qpOASES   |
+============+============+===========+
| **PM**     | * Diamond  | * Diamond |
|            | * Circle   | * Circle  |
+------------+------------+-----------+
| **KM**     | * Diamond  |           |
|            | * Circle   |           |
+------------+------------+-----------+
| **DM**     | * Circles  |           |
+------------+------------+-----------+
| **FW**     | * Circles  |           |
+------------+------------+-----------+

- PM: point mass model
- KM: kinematic bicycle model
- DM: dynamic bicycle model
- FW: double track model

**Important notes:**

- | All combinations of solvers and models support variable max. input power. The **DM** and **FW** models do currently
  | not support variable friction between tires and ground.

- | The solver OSQP is running online in the velocity optimization algoritm. OSQP is therefore not provided as a
  | benchmark solver as its outputs are already given in the logs. Still, the logged input data can be used to rerun
  | the first SQP (OSQP)-iteration to detect, e.g., infeasibility of the given problem.

As an example, the optimized velocity (OSQP) is plotted at the top of this page together with the solutions by
different solvers and vehicle
dynamics model (IPOPT + doulbe track model and qpOASES + point mass model in this case),
that are calculated during debugging (depending on the chosen options above). Plots for the driving force, motor power,
slack variables and combined accelerations are visualized:

Solver configurations
---------------------

The solver configurations can be selected in the sol_options dictionary.

.. list-table:: Visualization Options (Default values in brackets)
   :widths: 25 10 65
   :header-rows: 1

   * - Name
     - Value
     - Description
   * - Model
     - PM/KM/DM/FW
     - Select the vehicle dynamics model.
   * - Solver
     - IPOPT/qpOASES
     - Select between the solvers IPOPT (IP) and qoOASES (Active Set) to be compared to the OSQP (ADMM) solution.
   * - Friction
     - Circle/Diamond
     - | Select between the model for the combined acceleration limitaiton for PM or KM. DM and FW have Kamm
       | circles.
   * - VarFriction
     - True/False
     - Choose if the optimization problem is solved with a variable friction potential along the track.
   * - VarPower
     - True/False
     - Choose if a variable power constraint is used to solve the optimization problem.
   * - Slack
     - True/False
     - | Choose if slack variables are used in the optimization (True) or not (False). Only available for the PM and
       | KM in combination with the solver IPOPT.

In the code below, two configurations are set to solve the optimization problem and compare the OSQP-solution to. |br|
Solver 1 contains the point-mass model (FW) as the vehicle dynamics model, solved by IPOPT.The second solver contains
a PM model, where qpOASES is used to solve the problem.

.. code-block:: python

    sol_options = {'solver1': {'Model': "FW",               # PM (Pointmass), KM (Kinematic Single Track Model),
                                                            # DM (Dynamic Single Track Model), FW (Double Track Model)
                               'Solver': "IPOPT",           # IPOPT, qpOASES
                               'Friction': "Circle",        # Circle, Diamond (only for PM and KM, rest have Circles)
                               'VarFriction': True,         # Variable friction: True, False
                               'VarPower': False,           # Variable power: True, False
                               'Slack': True,               # Usage of slack variables on comb. acceleration (only
                                                            # reasonable on simple models like PM and KM): Keep True!
                               }
                   }
                   'solver2': {'Model': "PM",               # PM (Pointmass), KM (Kinematic Single Track Model),
                                                            # DM (Dynamic Single Track Model), FW (Double Track Model)
                               'Solver': "qpOASES",         # IPOPT, qpOASES
                               'Friction': "Diamond",       # Circle, Diamond (only for PM and KM, rest have Circles)
                               'VarFriction': True,         # Variable friction: True, False
                               'VarPower': False,           # Variable power: True, False
                               'Slack': True,               # Usage of slack variables on comb. acceleration (only
                                                            # reasonable on simple models like PM and KM): Keep True!
                               }
                   }
