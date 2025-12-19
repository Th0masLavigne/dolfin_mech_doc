#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Operator import Operator

################################################################################

class HyperIncompressibilityOperator(Operator):
    """
    Operator enforcing the incompressibility constraint in a hyperelastic 
    framework.

    This operator is the counterpart to the :class:`HyperHydrostaticPressureOperator`. 
    In a mixed displacement-pressure formulation, it assembles the equation 
    governing the Lagrange multiplier (pressure) to enforce that the material 
    is volume-preserving.

    The residual contribution is:

    .. math::
        \mathcal{R}_{incomp} = - \int_{\Omega_0} (J - 1) \delta P \, d\Omega_0

    Where:
        - :math:`J = \det(\mathbf{F})` is the volume ratio (Jacobian).
        - :math:`J - 1 = 0` is the strong form of the incompressibility constraint.
        - :math:`\delta P` is the test function associated with the pressure field.

    Attributes:
        kinematics (Kinematics): Kinematic variables providing the Jacobian :math:`J`.
        measure (dolfin.Measure): Integration measure (typically ``dx``).
        res_form (UFL form): The resulting residual variational form.
    """
    def __init__(self,
            kinematics,
            P_test,
            measure):
        """
        Initializes the HyperIncompressibilityOperator.

        :param kinematics: Kinematics object providing the Jacobian.
        :type kinematics: dmech.Kinematics
        :param P_test: Test function associated with the pressure field (Lagrange multiplier).
        :type P_test: dolfin.TestFunction
        :param measure: Dolfin measure for domain integration.
        :type measure: dolfin.Measure
        """
        self.kinematics = kinematics
        self.measure    = measure

        self.res_form = - (self.kinematics.J - 1) * P_test * self.measure
