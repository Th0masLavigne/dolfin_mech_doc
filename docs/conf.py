import os
import sys
from unittest.mock import MagicMock
import sphinx_rtd_theme

# Ajoute la racine du projet pour que Sphinx trouve dolfin_mech
sys.path.insert(0, os.path.abspath('..'))

# Mock des dépendances qui ne peuvent pas être installées sur GitHub (FEniCS + vos libs)
MOCK_MODULES = [
'dolfin', 
    'fenics', 
    'ufl',                 # Essential for FEniCS math expressions
    'numpy',               # Used in Kinematics.py
    'myPythonLibrary', 
    'myVTKPythonLibrary', 
    'vtkpython_cbl',
    'vtk'
]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()

# Configuration de base
project = 'dolfin_mech'
copyright = '2025, Martin Genet'
author = 'Martin Genet'

extensions = [
    'sphinx.ext.autodoc',     # Extraction des docstrings
    'sphinx.ext.napoleon',    # Support du format Google/NumPy
    'sphinx.ext.viewcode',    # Lien vers le code source
    'sphinx.ext.mathjax',     # Rendu des équations LaTeX
    'sphinx_rtd_theme',
]

# Thème visuel
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = [] # Leave empty for now if you don't have custom CSS

# Empêcher Sphinx de planter sur les erreurs d'importation mineures
autodoc_inherit_docstrings = True

add_module_names = False # Prevents showing the full path (dolfin_mech.Kinematics) in titles

html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 3,
}