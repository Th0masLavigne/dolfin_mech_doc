Getting Started
===============

**dolfin_mech** simplifies the implementation of complex mechanics problems. Below are the basic steps to set up a simulation.

Prerequisites
-------------
Ensure you have a working FEniCS (2019.1.0) environment[cite: 17, 18]. 

.. code-block:: bash

   conda activate dolfin_mech

Basic Workflow
--------------
Most simulations follow this pattern:

1. **Define Geometry**: Load a mesh or generate one using built-in scripts like ``run_Ball_Mesh.py``.
2. **Select Material**: Choose from Elastic, Inelastic, or Porous models[cite: 1, 13, 16].
3. **Define Operators**: Add physics (Inertia, Hyperelasticity, Darcy Flow)[cite: 4, 13].
4. **Solve**: Use the ``NonlinearSolver`` to find the solution.

First Example: Hyperelastic Ball
--------------------------------
To run your first simulation of a hyperelastic ball under loading, use:

.. code-block:: bash

   python -m dolfin_mech.run_Ball_Hyperelasticity

This example demonstrates the core ``Problem_Hyperelasticity`` class and how to apply surface pressure.