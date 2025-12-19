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
from .Material import Material

################################################################################

class ElasticMaterial(Material):
    """
    Base class for all elastic material models.

    This class serves as an abstraction layer for both linearized elasticity 
    (e.g., Hooke) and finite-strain hyperelasticity (e.g., Neo-Hookean, 
    Mooney-Rivlin). It inherits from :py:class:`Material` 
    to provide access to unified parameter conversion methods.

    Derived classes are expected to implement specific strain energy density 
    functions :math:`\\Psi` or stress-strain relationships.
    """
    pass
