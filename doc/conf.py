# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import pathlib
import sys

# Add paths to the code to the sphinx system path here
# Assume that the root of the python code is one level up from this file location
sys.path.insert(0, pathlib.Path(__file__).parents[1].joinpath('preprocessing').resolve().as_posix())
sys.path.insert(0, pathlib.Path(__file__).parents[1].joinpath('mne/lib64/python3.8/site-packages').resolve().as_posix())
print(sys.path)

project = 'eegfh'
copyright = '2023, AM'
author = 'AM'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc','sphinx.ext.autosummary', 
		'myst_parser',			# Use .md files as sources
		'sphinx.ext.napoleon',		# Use google or numpy doc strings instead of original
						# sphinx/rst doc strings
		]
autodoc_default_options = {
    'members': True		# This adds :members: option to automodule, autoclass by default
}
napoleon_custom_sections = [('Returns', 'params_style')]	# This allows describing multiple return values
								# and their types in 'Returns' section
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
add_module_names = False	# Do not prepend the module name to function descriptions


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

