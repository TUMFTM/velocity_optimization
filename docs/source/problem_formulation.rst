Problem formulation
===================

Basically, the code in this python package solves an optimization problem with the following objective function:

    .. math::
        J(x(s)) = &\int_{0}^{s_\mathrm{f}}{\frac{1}{v(s)}\mathrm{d}s} +
        \frac{\rho_{\mathrm{j}}}{s_\mathrm{f}}\int_{0}^{s_\mathrm{f}}{\left( \frac{\mathrm{d}^2 v(s)}{\mathrm{d}^2t}\right)^2} \mathrm{d}s + \\
        &\frac{\rho_{\mathrm{\epsilon,l}}}{s_\mathrm{f}} \int_{0}^{s_\mathrm{f}}{\epsilon(s)}\mathrm{d}s  +\frac{\rho_{\mathrm{\epsilon,q}}}{s_\mathrm{f}} \int_{0}^{s_\mathrm{f}}{\epsilon^2(s)}\mathrm{d}s.

Here, :math:`v(s)` denots a space-dependent velocity profile. This velocity profile :math:`v(s)` is maximized as the inverse of it
is minimized in the first term. The second term penalizes the vehicle's jerk, which is approximated by the second
derivative of the velocity profile. The third and fourth term add the optimization problem's slack variables
:math:`\epsilon(s)` to the
objective function. These are necessary to keep subsequent velocity optimization problems feasible as they soften
some of the hard constraints. The parameters :math:`\rho_i` can be used to tune the contribution of the single terms.

The detailed mathematical problem formulation behind the code and further explanations can be found in our publication.
The preprint of this publication can be found on arXiv:

    `https://arxiv.org/abs/2012.13586 <https://arxiv.org/abs/2012.13586>`_

The paper will soon be available on IEEE eXplore and in the printed version of the IEEE Transactions on Intelligent Vehicles:

    https://doi.org/10.1109/TIV.2020.3047858
