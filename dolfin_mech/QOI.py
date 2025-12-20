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

################################################################################

class QOI():
    """
    Class representing a Quantity of Interest (QOI).

    A QOI is a scalar value computed at the end of each time/load step. It is 
    used for post-processing and analysis, often representing global aggregates 
    (integrals) or specific point values.

    Common examples include:
    - **Global Integrals:** Total volume, total strain energy, homogenized stress.
    - **Surface Integrals:** Reaction forces, fluxes.
    - **Point Values:** Displacement at a specific node, pressure at a sensor location.

    

    The class handles the calculation method (assembly vs. direct evaluation) 
    and optional normalization or time-derivative scaling.
    """
    def __init__(self,
            name,
            expr=None,
            expr_lst=None,
            norm=1.,
            constant=0.,
            divide_by_dt=False,
            form_compiler_parameters={},
            point=None,
            update_type="assembly"):
        """
        Initializes a Quantity of Interest.

        :param name: The identifier for the QOI (e.g., "vol", "sigma_XX").
        :type name: str
        :param expr: The UFL form or expression to evaluate (for single-step or consistent definitions).
        :param expr_lst: A list of UFL forms, one for each load step (for multi-step simulations where the definition changes).
        :param norm: A scaling factor to divide the result by (e.g., total volume for homogenization).
        :type norm: float
        :param constant: A constant offset added to the result.
        :type constant: float
        :param divide_by_dt: If True, divides the result by the time step ``dt`` (useful for rates/velocities).
        :type divide_by_dt: bool
        :param form_compiler_parameters: Compiler options for the FEniCS assembler.
        :type form_compiler_parameters: dict
        :param point: The spatial coordinate for point-wise evaluation (used if ``update_type="direct"``).
        :type point: tuple/list
        :param update_type: The evaluation strategy:
            - ``"assembly"``: Performs a global FEM integration (assemble).
            - ``"direct"``: Evaluates a function at a specific point.
        :type update_type: str
        """

        self.name                     = name
        self.expr                     = expr
        self.expr_lst                 = expr_lst
        self.norm                     = norm
        self.constant                 = constant
        self.divide_by_dt             = divide_by_dt
        self.form_compiler_parameters = form_compiler_parameters
        self.point                    = point

        if (update_type == "assembly"):
            self.update = self.update_assembly
        elif (update_type == "direct"):
            self.update = self.update_direct



    def update_assembly(self, dt=None, k_step=None, expr=None):
        """
        Computes the QOI value by assembling (integrating) the UFL form over the mesh.

        This method supports both static definitions (single ``expr``) and 
        dynamic definitions (list of expressions in ``expr_lst``) where the 
        quantity being measured changes between steps.

        :param dt: Time step size (required if ``divide_by_dt`` is True).
        :param k_step: Current step index (used to select from ``expr_lst``).
        :param expr: Optional override for the expression to assemble.
        """
        # print(self.name)
        # print(self.expr)
        # print(self.form_compiler_parameters)
        if (self.expr is not None):
            self.value = dolfin.assemble(
                self.expr,
                form_compiler_parameters=self.form_compiler_parameters)
        else:
            if (k_step is None):
                self.value = dolfin.assemble(
                    self.expr_lst[0],
                    form_compiler_parameters=self.form_compiler_parameters)
            else:
                self.value = dolfin.assemble(
                    self.expr_lst[k_step - 1],
                    form_compiler_parameters=self.form_compiler_parameters)


        self.value += self.constant
        self.value /= self.norm

        if (self.divide_by_dt):
            assert (dt != 0),\
                "dt (="+str(dt)+") should be non zero. Aborting."
            self.value /= dt



    def update_direct(self, dt=None, k_step=None):
        """
        Computes the QOI value by directly evaluating a function at a specific point.

        This is faster than assembly and appropriate for tracking values at 
        nodes or specific spatial locations.

        :param dt: Time step size (required if ``divide_by_dt`` is True).
        :param k_step: Current step index (unused for direct updates generally).
        """
        self.value = self.expr(self.point)

        self.value += self.constant
        self.value /= self.norm

        if (self.divide_by_dt) and (dt is not None):
            self.value /= dt
