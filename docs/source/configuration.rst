Configuration
=============

The velocity planner is parametrized using some configuraiton scripts. Adapt the paths to your own configuration scripts
as shown in the `examples`-section of this documentation. An example configuration-script can be found in this package
in `velocity_optimization/params/sqp_config.ini`. Simply copy this file to any location on your machine. You may want
to adapt the following vehicle parameters:

+-------------+----------------------------------------+--------------------------------------------------------------------------------------------------+
| Parameter   | Description                            | Comment                                                                                          |
+=============+========================================+==================================================================================================+
| mass_t      | Vehicle mass [t]                       |                                                                                                  |
+-------------+----------------------------------------+--------------------------------------------------------------------------------------------------+
| P_max_kW    | Maximum vehicle power [kW]             |                                                                                                  |
+-------------+----------------------------------------+--------------------------------------------------------------------------------------------------+
| F_max_kN    | Maximum tractional force [kN]          |                                                                                                  |
+-------------+----------------------------------------+--------------------------------------------------------------------------------------------------+
| F_min_kN    | Maximum braking force [kN]             |                                                                                                  |
+-------------+----------------------------------------+--------------------------------------------------------------------------------------------------+
| ax_max_mps2 | Max. longitudinal acceleration [mps2]  | Won't be used if given by friction estimation module in your path planner in `veh_dynamics_info` |
+-------------+----------------------------------------+--------------------------------------------------------------------------------------------------+
| ay_max_mps2 | Max. lateral acceleration [mps2]       | Won't be used if given by friction estimation module in your path planner in `veh_dynamics_info` |
+-------------+----------------------------------------+--------------------------------------------------------------------------------------------------+
| c_res       | Air resistance coeff. [kg/m]           |                                                                                                  |
+-------------+----------------------------------------+--------------------------------------------------------------------------------------------------+
