#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Cécile Patte, 2019-2021                                              ###
###                                                                          ###
### INRIA, Palaiseau, France                                                 ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Material_Elastic import ElasticMaterial

################################################################################

class WbulkLungElasticMaterial(ElasticMaterial):
    """
    Class representing the bulk (volumetric) elastic energy contribution specifically 
    tailored for lung tissue modeling, based on solid volume fraction.

    This material model uses a logarithmic energy potential to penalize deviations 
    from the reference solid volume fraction :math:`\Phi_{s0}`.

    The strain energy density function is defined as:

    .. math::
        \Psi = \kappa \left( \\frac{\Phi_s}{\Phi_{s0}} - 1 - \ln\\left( \\frac{\Phi_s}{\Phi_{s0}} \\right) \\right)

    Where:
        - :math:`\kappa` is the bulk modulus (penalty parameter).
        - :math:`\Phi_s` is the current solid volume fraction.
        - :math:`\Phi_{s0}` is the reference solid volume fraction.

    Attributes:
        kappa (dolfin.Constant): Bulk modulus parameter.
        Psi (UFL expression): The calculated strain energy density.
        dWbulkdPhis (UFL expression): The derivative of the energy with respect to 
            the solid volume fraction :math:`\\frac{\partial \Psi}{\partial \Phi_s}`.
    """
    def __init__(self,
            Phis,
            Phis0,
            parameters):
        """
        Initializes the WbulkLungElasticMaterial.

        :param Phis: Current solid volume fraction.
        :type Phis: dolfin.Function or dolfin.Expression
        :param Phis0: Reference solid volume fraction.
        :type Phis0: dolfin.Function, dolfin.Constant, or float
        :param parameters: Dictionary containing material parameters. Must include 'kappa'.
        :type parameters: dict
        :raises AssertionError: If 'kappa' is not present in the parameters.
        """
        assert ('kappa' in parameters)
        self.kappa = dolfin.Constant(parameters['kappa'])

        Phis = dolfin.variable(Phis)
        self.Psi = self.kappa * (Phis/Phis0 - 1 - dolfin.ln(Phis/Phis0))
        self.dWbulkdPhis = dolfin.diff(self.Psi, Phis)
