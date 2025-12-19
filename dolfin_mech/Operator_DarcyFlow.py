#coding=utf8

################################################################################
###                                                                          ###
### Created by Haotian XIAO, 2024-2027                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Operator import Operator

################################################################################

class DarcyFlowOperator(Operator):
    """
    Operator representing Darcy's law for fluid flow through a porous medium 
    undergoing large deformations.

    This operator assembles the residual for the fluid diffusion equation. It 
    accounts for the change in permeability as the medium deforms (Piola 
    transformation of the permeability tensor).

    The residual form is:

    .. math::
        \mathcal{R} = \int_{\Omega} \\rho_l \left( \mathbf{k}_l \mathbf{F}^{-1} \\nabla p \\right) \cdot \\nabla p_{test} \, dx - \int_{\Gamma_{in}} \Theta_{in} p_{test} \, ds + \int_{\Gamma_{out}} \Theta_{out} p_{test} \, ds

    Where:
        - :math:`\mathbf{k}_l = \\frac{1}{J} \mathbf{F} \mathbf{K}_l \mathbf{F}^T` is the spatial permeability.
        - :math:`\mathbf{K}_l` is the reference (material) permeability.
        - :math:`\\rho_l` is the fluid density.
        - :math:`\Theta` represents source/sink terms at the boundaries.

    Attributes:
        kinematics (Kinematics): Kinematic variables (F, J).
        res_form (UFL form): The resulting residual variational form.
        k_l (UFL expression): The spatial permeability tensor.
    """
    def __init__(self,
                 kinematics,
                 p,
                 p_test,
                 K_l,
                 rho_l,
                 dx,
                 dx_in,
                 dx_out,
                 Theta_in=None,
                 Theta_out=None):
        """
        Initializes the DarcyFlowOperator.

        :param kinematics: Kinematics object for deformation handling.
        :param p: Fluid pressure trial function.
        :param p_test: Fluid pressure test function.
        :param K_l: Permeability tensor in the reference configuration.
        :param rho_l: Fluid density.
        :param dx: Global measure for the domain.
        :param dx_in: Measure for the inlet subdomain.
        :param dx_out: Measure for the outlet subdomain.
        :param Theta_in: Flux at the inlet. Defaults to Constant(0.0).
        :type Theta_in: dolfin.Constant or float, optional
        :param Theta_out: Flux at the outlet. Defaults to Constant(0.0).
        :type Theta_out: dolfin.Constant or float, optional
        """   
        assert dx is not None, "You must provide a global measure dx."
        assert dx_in is not None and dx_out is not None, "You must provide inlet and outlet subdomain measures."

        self.measure = dx  # typically dx(0) or full domain
        self.kinematics = kinematics

        if Theta_in is None:
            Theta_in = dolfin.Constant(0.0)
        if Theta_out is None:
            Theta_out = dolfin.Constant(0.0)
        
        grad_p = dolfin.grad(p)
        grad_p_test = dolfin.grad(p_test)

        F = self.kinematics.F
        J = self.kinematics.J
        # K_l : permeability tensor in reference config (material)
        k_l = (1.0 / J) * F * K_l * F.T  # current configuration permeability
        self.K_l = K_l  # keep reference permeability for output
        self.k_l = k_l  # keep current permeability for output
        self.J = J

        grad_p = dolfin.grad(p)
        grad_p_test = dolfin.grad(p_test)

        # --- Darcy flow residual (standard diffusion-like form) ---
        self.res_form = rho_l * dolfin.inner(k_l * dolfin.inv(kinematics.F) * grad_p, grad_p_test) * dx
        if Theta_in != 0.0:
            self.res_form -= Theta_in * p_test * dx_in
        if Theta_out != 0.0:
            self.res_form += Theta_out * p_test * dx_out





class PlFieldOperator(Operator):
    """
    Operator to enforce the fluid pressure field :math:`p_l` within the 
    variational formulation.

    This is typically used in mixed formulations to couple the pressure 
    unknown with other fields (like porosity or displacement).

    Attributes:
        res_form (UFL form): Residual term :math:`\int p_l \cdot \phi_{test} \, dx`.
    """
    def __init__(self,
                 pl,
                 unknown_porosity_test,
                 measure):
        """
        Initializes the PlFieldOperator.

        :param pl: Fluid pressure variable.
        :param unknown_porosity_test: Test function for the porosity field.
        :param measure: Integration measure.
        """
        self.measure = measure
        self.res_form = dolfin.inner(pl, unknown_porosity_test) * self.measure

class WbulkPoroFlowOperator(Operator):
    """
    Operator representing the volumetric coupling between the solid skeleton 
    and fluid pressure in a poroelastic medium.

    It combines the elastic bulk energy of the porous skeleton (lung tissue 
    model) with the work done by the fluid pressure :math:`p_l` during 
    volumetric changes.

    The residual includes:
        1. The stress contribution from the fluid pressure.
        2. The constitutive response of the solid volume fraction :math:`\Phi_s`.

    Attributes:
        solid_material (WbulkLungElasticMaterial): The underlying solid phase material.
        material (PorousElasticMaterial): The scaled porous material.
        res_form (UFL form): The combined residual form.
    """
    def __init__(self,
            kinematics,
            U,
            U_test,
            Phis0,
            Phis,
            Phis_test,
            material_parameters,
            material_scaling,
            measure,
            pl
            ):  # new input
        """
        Initializes the WbulkPoroFlowOperator.

        :param kinematics: Kinematics object.
        :param U: Displacement field.
        :param U_test: Displacement test function.
        :param Phis0: Initial solid volume fraction.
        :param Phis: Current solid volume fraction.
        :param Phis_test: Solid volume fraction test function.
        :param material_parameters: Dict of parameters for the lung material.
        :param material_scaling: Scaling type ('no' or 'linear').
        :param pl: Fluid pressure.
        """
        self.kinematics = kinematics
        self.solid_material = dmech.WbulkLungElasticMaterial(
            Phis=Phis,
            Phis0=Phis0,
            parameters=material_parameters)
        self.material = dmech.PorousElasticMaterial(
            solid_material=self.solid_material,
            scaling=material_scaling,
            Phis0=Phis0)
        self.measure = measure

        dE_test = dolfin.derivative(
            self.kinematics.E, U, U_test)

        self.res_form =  dolfin.inner(
            pl * self.kinematics.J * self.kinematics.C_inv,
            dE_test) * self.measure

        self.res_form += self.material.dWbulkdPhis * Phis_test * self.measure