#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Strawberry Fields documentation build configuration file, created by
# sphinx-quickstart on Fri Sep  8 14:44:21 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys, os, re
from unittest.mock import MagicMock, PropertyMock

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('_ext'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath('.')), 'doc'))


class Mock(MagicMock):
    __name__ = 'foo'

    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


sys.modules["tensorflow"] = Mock(__version__="1.3")

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '1.5'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    # 'sphinx.ext.imgmath',
    'sphinx.ext.napoleon',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex',
    'edit_on_github',
    'sphinx_autodoc_typehints',
    'nbsphinx',
    'sphinx_gallery.gen_gallery'
]

from glob import glob
import shutil
import warnings

mathjax_path = "https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

# nbsphinx settings

exclude_patterns = ['_build', '**.ipynb_checkpoints']
nbsphinx_execute = 'never'
nbsphinx_epilog = """
.. note:: :download:`Click here <../../{{env.docname}}.ipynb>` to download this gallery page as an interactive Jupyter notebook.
"""

sphinx_gallery_conf = {
    # path to your example scripts
    'examples_dirs': '../examples_gbs',
    # path where to save gallery generated examples
    'gallery_dirs': 'tutorials_gbs',
    # execute files that match the following filename pattern,
    # and skip those that don't. If the following option is not provided,
    # all example scripts in the 'examples_dirs' folder will be skiped.
    'filename_pattern': r'run',
    # first notebook cell in generated Jupyter notebooks
    'first_notebook_cell': ("# This cell is added by sphinx-gallery\n"
                            "# It can be customized to whatever you like\n"
                            "%matplotlib inline"),
    # thumbnail size
    'thumbnail_size': (400, 400),
    'capture_repr': (),
}

# Remove warnings that occur when generating the the tutorials
warnings.filterwarnings("ignore", category=UserWarning, message=r"Matplotlib is currently using agg")
warnings.filterwarnings("ignore", category=FutureWarning, message=r"Passing \(type, 1\) or '1type' as a synonym of type is deprecated.+")

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates', 'xanadu_theme']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'Strawberry Fields'
copyright = """Nathan Killoran, Josh Izaac, Nicolás Quesada, Ville Bergholm, Matthew Amy, and Christian Weedbrook. <br>
"Strawberry Fields: A Software Platform for Photonic Quantum Computing", Quantum, 3, 129 (2019).<br>
&copy; Copyright 2019, Xanadu Quantum Technologies Inc."""
author = 'Xanadu Inc.'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The full version, including alpha/beta/rc tags.
import strawberryfields as sf
release = sf.__version__

# The short X.Y version.
version = re.match(r'^(\d+\.\d+)', release).expand(r'\1')

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = '%Y-%m-%d'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build']

# The reST default role (used for this markup: `text`) to use for all
# documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
show_authors = True

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
#keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# html_theme = 'nature'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.
#html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = None

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = '_static/favicon.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
#html_extra_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
#html_sidebars = {
#    '**': [
#        'about.html',
#        'navigation.html',
#        'relations.html',  # needs 'show_related': True theme option to display
#        'searchbox.html',
#        'donate.html',
#    ]
#}
html_sidebars = {
    '**' : [
        'logo-text.html',
        'searchbox.html',
        'globaltoc.html',
        # 'sourcelink.html'
    ]
}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Language to be used for generating the HTML full-text search index.
# Sphinx supports the following languages:
#   'da', 'de', 'en', 'es', 'fi', 'fr', 'h', 'it', 'ja'
#   'nl', 'no', 'pt', 'ro', 'r', 'sv', 'tr'
#html_search_language = 'en'

# A dictionary with options for the search language support, empty by default.
# Now only 'ja' uses this config value
#html_search_options = {'type': 'default'}

# The name of a javascript file (relative to the configuration directory) that
# implements a search results scorer. If empty, the default will be used.
#html_search_scorer = 'scorer.js'

# Output file base name for HTML help builder.
htmlhelp_basename = 'Strawberryfieldsdoc'

# # -- Xanadu theme ---------------------------------------------------------
html_theme = 'xanadu_theme'
html_theme_path = ['.']

# Register the theme as an extension to generate a sitemap.xml
# extensions.append("guzzle_sphinx_theme")

# xanadu theme options (see theme.conf for more information)
html_theme_options = {
    # Set the path to a special layout to include for the homepage
    # "homepage": "special_index.html",

    # Set the name of the project to appear in the left sidebar.
    "project_nav_name": "Strawberry Fields",
    "project_logo": "_static/strawberry_fields.png",
    "touch_icon": "_static/logo_new.png",
    "touch_icon_small": "_static/logo_new_small.png",
    "large_toc": True,

    # Set GA account ID to enable tracking
    "google_analytics_account": "UA-116279123-2",

    # colors
    "navigation_button": "#b13a59",
    "navigation_button_hover": "#712b3d",
    "toc_caption": "#b13a59",
    "toc_hover": "#b13a59",
    "table_header_bg": "#ffdce5",
    "table_header_border": "#b13a59",
    "download_button": "#b13a59",

    # gallery options
    "github_repo": "XanaduAI/strawberryfields",
    "gallery_dirs": sphinx_gallery_conf['gallery_dirs']
}

edit_on_github_project = 'XanaduAI/strawberryfields'
edit_on_github_branch = 'master/doc'


# the order in which autodoc lists the documented members
autodoc_member_order = 'bysource'

# inheritance_diagram graphviz attributes
inheritance_node_attrs = dict(color='lightskyblue1', style='filled')


from custom_directives import IncludeDirective, GalleryItemDirective, CustomGalleryItemDirective

def setup(app):
    app.add_directive('includenodoc', IncludeDirective)
    app.add_directive('galleryitem', GalleryItemDirective)
    app.add_directive('customgalleryitem', CustomGalleryItemDirective)
    app.add_stylesheet('xanadu_gallery.css')
