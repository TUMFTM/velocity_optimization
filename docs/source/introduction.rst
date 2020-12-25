Introduction
============

.. image:: vel.png
   :width: 600

``velocity_optimization`` is a python package that optimizes the velocity for a vehicle (minimizes travelling time)
with respect to necessary physical constraints, e.g., maximum available power, combined longitudinal and lateral
acceleration, maximum available driving/braking force, calculation time, ... . The mathematical method behind is called
Sequential Quadratic Programming. The single Quadratic Problems are solved by `OSQP
<https://osqp.org/>`_.
The package was developed and tested to power the race vehicle of the
`Technical University of Munich <https://www.mw.tum
.de/en/ftm/main-research/vehicle-dynamics-and-control-systems/roborace-autonomous-motorsport/>`_ in the `Roborace
<https://roborace.com/>`_ competition.
