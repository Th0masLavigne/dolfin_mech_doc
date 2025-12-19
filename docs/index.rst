dolfin_mech
===========

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8010870.svg?style=flat-square
   :target: https://doi.org/10.5281/zenodo.8010870
.. image:: https://img.shields.io/pypi/v/dolfin-mech.svg?style=flat-square
   :target: https://pypi.org/project/dolfin-mech
.. image:: https://img.shields.io/github/license/mgenet/dolfin_mech
   :target: https://github.com/mgenet/dolfin_mech/blob/master/LICENSE
.. image:: https://static.pepy.tech/badge/dolfin-mech
   :target: https://pepy.tech/projects/dolfin-mech

**dolfin_mech** is a Python library for computational mechanics based on FEniCS. The library has notably been used in:

* `Genet (2019). A relaxed growth modeling framework for controlling growth-induced residual stresses. Clinical Biomechanics. <https://doi.org/10.1016/j.clinbiomech.2019.08.015>`_
* `Álvarez-Barrientos, Hurtado & Genet (2021). Pressure-driven micro-poro-mechanics: A variational framework for modeling the response of porous materials. International Journal of Engineering Science. <https://doi.org/10.1016/j.ijengsci.2021.103586>`_
* `Patte, Genet & Chapelle (2022). A quasi-static poromechanical model of the lungs. Biomechanics and Modeling in Mechanobiology. <https://doi.org/10.1007/s10237-021-01547-0>`_
* `Patte, Brillet, Fetita, Gille, Bernaudin, Nunes, Chapelle & Genet (2022). Estimation of regional pulmonary compliance in idiopathic pulmonary fibrosis based on personalized lung poromechanical modeling. Journal of Biomechanical Engineering. <https://doi.org/10.1115/1.4054106>`_
* `Tueni, Allain & Genet (2023). On the structural origin of the anisotropy in the myocardium: Multiscale modeling and analysis. Journal of the Mechanical Behavior of Biomedical Materials. <https://doi.org/10.1016/j.jmbbm.2022.105600>`_
* `Laville, Fetita, Gille, Brillet, Nunes, Bernaudin & Genet (2023). Comparison of optimization parametrizations for regional lung compliance estimation using personalized pulmonary poromechanical modeling. Biomechanics and Modeling in Mechanobiology. <https://doi.org/10.1007/s10237-023-01691-9>`_
* `Peyraut & Genet (2024). A model of mechanical loading of the lungs including gravity and a balancing heterogeneous pleural pressure. Biomechanics and Modeling in Mechanobiology. <https://doi.org/10.1007/s10237-024-01876-w>`_
* `Peyraut & Genet (2025). Finite strain formulation of the discrete equilibrium gap principle: application to direct parameter estimation from large full-fields measurements. Comptes Rendus Mécanique. <https://doi.org/10.5802/crmeca.279>`_
* `Manoochehrtayebi, Bel-Brunon & Genet (2025). Finite strain micro-poro-mechanics: Formulation and compared analysis with macro-poro-mechanics. International Journal of Solids and Structures. <https://doi.org/10.1016/j.ijsolstr.2025.113354>`_
* `Peyraut & Genet (2025). Inverse Uncertainty Quantification for Personalized Biomechanical Modeling: Application to Pulmonary Poromechanical Digital Twins. Journal of Biomechanical Engineering. <https://doi.org/10.1115/1.4068578>`_
* `Manoochehrtayebi, Genet & Bel-Brunon (2025). Micro-poro-mechanical modeling of lung parenchyma: Theoretical modeling and parameters identification. Journal of Biomechanical Engineering. <https://doi.org/10.1115/1.4070036>`_

Installation
------------

A working installation of `FEniCS <https://fenicsproject.org>`_ (version 2019.1.0) is required to run ``dolfin_mech``.

To setup a system, the simplest is to use `conda <https://conda.io>`_: first install `miniconda <https://docs.conda.io/projects/miniconda/en/latest>`_ (note that for Microsoft Windows machines you first need to install WSL, the `Windows Subsystem for Linux <https://learn.microsoft.com/en-us/windows/wsl/install>`_, and then install miniconda for linux inside the WSL), and then install the necessary packages:

.. code-block:: bash

   conda create -y -c conda-forge -n dolfin_mech fenics=2019.1.0 matplotlib=3.5 meshio=5.3 mpi4py=3.1.3 numpy=1.23 pandas=1.3 pip python=3.10 vtk=9.2
   conda activate dolfin_mech

Now, if you only need to use the library, you can install it with:

.. code-block:: bash

   pip install dolfin_mech

But if you need to develop within the library, you need to install an editable version of the sources:

.. code-block:: bash

   git clone https://github.com/mgenet/dolfin_mech.git
   pip install -e dolfin_mech/.


.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   Introduction <self>
   Getting Started <getting_started>

.. toctree::
   :maxdepth: 1
   :caption: Core API
   :hidden:

   Kinematics <dolfin_mech.Kinematics>
   Kinematics Inverse <dolfin_mech.Kinematics_Inverse>
   Kinematics Linearized <dolfin_mech.Kinematics_Linearized>
   Constraint <dolfin_mech.Constraint>
   Loading <dolfin_mech.Loading>
   Problem <dolfin_mech.Problem>
   Step <dolfin_mech.Step>
   Nonlinear Solver <dolfin_mech.NonlinearSolver>
   Time Integrator <dolfin_mech.TimeIntegrator>

.. toctree::
   :maxdepth: 1
   :caption: Materials
   :hidden:

   Material Base <dolfin_mech.Material>
   Elastic Base <dolfin_mech.Material_Elastic>
   Exponential Neo-Hookean Elastic <dolfin_mech.Material_Elastic_ExponentialNeoHookean>
   Exponential Ogden Ciarlet Geymonat Elastic <dolfin_mech.ExponentialOgdenCiarletGeymonatElasticMaterial>
   Hooke Elastic <dolfin_mech.Material_Elastic_Hooke>
   Inelastic Material <dolfin_mech.Material_Inelastic>
   Kirchhoff <dolfin_mech.Material_Elastic_Kirchhoff>
   Neo-Hookean <dolfin_mech.Material_Elastic_NeoHookean>
   Porous <dolfin_mech.Material_Elastic_Porous>

.. toctree::
   :maxdepth: 1
   :caption: Mesh
   :hidden:




.. toctree::
   :maxdepth: 1
   :caption: Operators
   :hidden:

   Operator <dolfin_mech.Operator>
   Hyper-Elasticity <dolfin_mech.Operator_HyperElasticity>
   Darcy Flow <dolfin_mech.Operator_DarcyFlow>
   Inertia <dolfin_mech.Operator_Inertia>
   Pressure Loading <dolfin_mech.Operator_Loading_SurfacePressure>


.. toctree::
   :maxdepth: 1
   :caption: Utilities
   :hidden:

   Field of Interest (FOI) <dolfin_mech.FOI>
   Quantity of Interest (QOI) <dolfin_mech.QOI>
   XDMF File <dolfin_mech.XDMFFile>
   Compute Error <dolfin_mech.compute_error>
   Mesh Function Expression (C++) <dolfin_mech.Expression_MeshFunction_cpp>


.. toctree::
   :maxdepth: 1
   :caption: Examples
   :hidden:

   Ball Hyperelasticity <dolfin_mech.run_Ball_Hyperelasticity>
   Disc Hyperelasticity <dolfin_mech.run_Disc_Hyperelasticity>
   Heart Slice <dolfin_mech.run_Heart_Slice_Hyperelasticity>
   Hollow Box MicroPoro <dolfin_mech.run_HollowBox_MicroPoroHyperelasticity>
   Poro-flow <dolfin_mech.run_Poroflow>
   Rivlin Cube Poro <dolfin_mech.run_RivlinCube_PoroHyperelasticity>

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`