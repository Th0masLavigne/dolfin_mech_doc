import os
import sys
from unittest.mock import MagicMock

# 1. Pointer vers la racine pour trouver dolfin_mech
sys.path.insert(0, os.path.abspath('../../'))

# 2. Mocker les dépendances complexes (FEniCS et vos libs privées)
# Cela permet à Sphinx d'importer votre code sans exécuter les imports dolfin
MOCK_MODULES = [
    'dolfin', 
    'fenics', 
    'myPythonLibrary', 
    'myVTKPythonLibrary', 
    'vtkpython_cbl'
]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()

# 3. Extensions standard
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
]

# 4. Utiliser le thème ReadTheDocs (installé dans le workflow)
html_theme = 'sphinx_rtd_theme'