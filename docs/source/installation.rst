Installation
============

General
-------

Set up a virtual envrionment using `virtualenv` and install the requirements given in `requirements.txt`. Insert a
name of your choice into `<your-venv-name>`.

.. code-block:: bash

    python3 -m venv <your-venv-name>

Now, activate the virtual environment and install the `velocity-optimization` package

.. code-block:: bash

    source <your-venv-name>/bin/activate
    pip install velocity-optimization>=1.0.0

To run the code
---------------
To run the vlocity-optimization algorithm, copy the folders `params <https://github
.com/TUMFTM/velocity_optimization/tree/master/velocity_optimization>`_  and `inputs <https://github
.com/TUMFTM/velocity_optimization/tree/master/velocity_optimization>`_  to any location on your machine and
adapt the paths in the code to these folders accordingly!

To run the debug tool
---------------------
Make sure you don't encounter any WARNINGS when starting the debug tool! If you get warnings, install the missing
python packages (like e.g., `CasADi>=3.5.1` or `tikzplotlib>=0.9.1`)!
