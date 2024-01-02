# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Xinference'
copyright = '2023, Xorbits Inc.'
author = 'xorbitsai'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_tabs.tabs",
    "sphinx_design",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# i18n
locale_dirs = ["locale/"]  # path is example but recommended.
gettext_compact = False  # optional


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'
html_title = "Xinference"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Define the json_url for our version switcher.
version_match = os.environ.get("READTHEDOCS_LANGUAGE")
json_url = "https://inference.readthedocs.io/en/latest/_static/switcher.json"
if not version_match:
    version_match = 'en'

html_theme_options = {
    "show_toc_level": 2,
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/xorbitsai/inference",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
    "navbar_align": "content",  # [left, content, right] For testing that the navbar items align properly
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_center": ["navbar-nav"],
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },
}


if version_match != 'zh-cn':
    html_theme_options['icon_links'].extend([{
        "name": "Slack",
        "url": "https://join.slack.com/t/xorbitsio/shared_invite/zt-1o3z9ucdh-RbfhbPVpx7prOVdM1CAuxg",
        "icon": "fa-brands fa-slack",
        "type": "fontawesome",
    },
    {
        "name": "Twitter",
        "url": "https://twitter.com/xorbitsio",
        "icon": "fa-brands fa-twitter",
        "type": "fontawesome",
    }])
else:
    html_theme_options['icon_links'].extend([{
        "name": "WeChat",
        "url": "https://xorbits.cn/assets/images/wechat_pr.png",
        "icon": "fa-brands fa-weixin",
        "type": "fontawesome",
    },
    {
        "name": "Zhihu",
        "url": "https://zhihu.com/org/xorbits",
        "icon": "fa-brands fa-zhihu",
        "type": "fontawesome",
    }])
    html_theme_options["external_links"] = [
        {"name": "产品官网", "url": "https://xorbits.cn/inference"},
    ]

html_favicon = "_static/favicon.svg"
