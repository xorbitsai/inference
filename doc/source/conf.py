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
import subprocess
import sys
# sys.path.insert(0, os.path.abspath('.'))

from pathlib import Path

from docutils import nodes

_doc_root = Path(__file__).resolve().parent.parent
if str(_doc_root) not in sys.path:
    sys.path.insert(0, str(_doc_root))
from i18n_locales import resolve_sphinx_language  # noqa: E402


# -- Project information -----------------------------------------------------

project = 'Xinference'
copyright = '2026, XINFERENCE HOLDINGS PTE. LTD.'
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
html_css_files = ["custom.css"]

json_url = "_static/switcher.json"

_SPHINX_LANGUAGE_TO_SWITCHER = {
    "en": "en",
    "zh_CN": "zh-cn",
    "zh-cn": "zh-cn",
    "zh_TW": "zh-tw",
    "zh-tw": "zh-tw",
    "ja": "ja",
    "ko": "ko",
    "de": "de",
    "fr": "fr",
    "es": "es",
    "it": "it",
    "pt_BR": "pt-br",
    "pt-br": "pt-br",
}

_EXTERNAL_LINKS_BY_LOCALE = {
    "zh-cn": {"name": "产品官网", "url": "https://xinference.cn"},
    "zh-tw": {"name": "產品官網", "url": "https://xinference.cn"},
    "ja": {"name": "公式サイト", "url": "https://xinference.io"},
    "ko": {"name": "공식 사이트", "url": "https://xinference.io"},
    "de": {"name": "Offizielle Website", "url": "https://xinference.io"},
    "fr": {"name": "Site officiel", "url": "https://xinference.io"},
    "es": {"name": "Sitio oficial", "url": "https://xinference.io"},
    "it": {"name": "Sito ufficiale", "url": "https://xinference.io"},
    "pt-br": {"name": "Site oficial", "url": "https://xinference.io"},
}
_DEFAULT_EXTERNAL_LINK = {"name": "Official Site", "url": "https://xinference.io"}

# PyData theme reads this directly; it is not translated via gettext catalogs.
_HEADER_DROPDOWN_TEXT_BY_LOCALE = {
    "en": "More",
    "zh-cn": "更多",
    "zh-tw": "更多",
    "ja": "その他",
    "ko": "더보기",
    "de": "Mehr",
    "fr": "Plus",
    "es": "Más",
    "it": "Altro",
    "pt-br": "Mais",
}

version_match = os.environ.get("READTHEDOCS_LANGUAGE") or "en"

_sphinx_language = resolve_sphinx_language(version_match)
if _sphinx_language:
    language = _sphinx_language


def _locale_po_files(locale: str) -> list[Path]:
    po_root = _doc_root / "source" / "locale" / locale / "LC_MESSAGES"
    if not po_root.is_dir():
        return []
    return sorted(po_root.rglob("*.po"))


def _needs_mo_compile(locale: str) -> bool:
    for po in _locale_po_files(locale):
        mo = po.with_suffix(".mo")
        if not mo.is_file() or mo.stat().st_mtime < po.stat().st_mtime:
            return True
    return False


def _compile_mo_catalog(locale: str) -> None:
    """Compile .po -> .mo when catalogs are missing or stale."""
    if not _locale_po_files(locale):
        return
    try:
        subprocess.run(
            [sys.executable, str(_doc_root / "build_i18n.py"), locale],
            cwd=str(_doc_root),
            check=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        print(
            f"[sphinx] warning: failed to compile locale/{locale} catalogs "
            f"({exc}); continuing with existing .mo files if any",
            flush=True,
        )


if _sphinx_language and _needs_mo_compile(_sphinx_language):
    _compile_mo_catalog(_sphinx_language)

if version_match == 'zh-cn' or _sphinx_language == "zh_CN":
    tags.add("zh_cn")


def _resolve_switcher_version(app):
    rtd_language = os.environ.get("READTHEDOCS_LANGUAGE")
    if rtd_language:
        return rtd_language
    current_language = getattr(app.config, "language", None) or "en"
    return _SPHINX_LANGUAGE_TO_SWITCHER.get(
        current_language, current_language.replace("_", "-").lower()
    )


html_theme_options = {
    "show_toc_level": 2,
    "header_links_before_dropdown": 7,
    "logo": {
        "image_light": "_static/xinference-logo-light.png",
        "image_dark": "_static/xinference-logo-dark.png",
        "alt_text": "Xinference",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/xorbitsai/inference",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "Telegram",
            "url": "https://t.me/+nCNpwmySwk9iYmI1",
            "icon": "fa-brands fa-telegram",
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
        "name": "Discord",
        "url": "https://discord.gg/Xw9tszSkr5",
        "icon": "fa-brands fa-discord",
        "type": "fontawesome",
    },
    {
        "name": "Twitter",
        "url": "https://twitter.com/xorbitsio",
        "icon": "fa-brands fa-twitter",
        "type": "fontawesome",
    }])
    html_theme_options["header_links_before_dropdown"] = 3
else:
    html_theme_options['icon_links'].extend([{
        "name": "Zhihu",
        "url": "https://zhihu.com/org/xorbits",
        "icon": "fa-brands fa-zhihu",
        "type": "fontawesome",
    }])

html_theme_options["external_links"] = [
    _EXTERNAL_LINKS_BY_LOCALE.get(version_match, _DEFAULT_EXTERNAL_LINK)
]
html_theme_options["header_dropdown_text"] = _HEADER_DROPDOWN_TEXT_BY_LOCALE.get(
    version_match, "More"
)

html_favicon = "_static/xinference-favicon.png"


def _apply_locale_theme_options(app, config):
    switcher_version = _resolve_switcher_version(app)
    config.html_theme_options.setdefault("switcher", {})["version_match"] = switcher_version
    config.html_theme_options["external_links"] = [
        _EXTERNAL_LINKS_BY_LOCALE.get(switcher_version, _DEFAULT_EXTERNAL_LINK)
    ]
    config.html_theme_options["header_dropdown_text"] = (
        _HEADER_DROPDOWN_TEXT_BY_LOCALE.get(switcher_version, "More")
    )
    if switcher_version == "zh-cn":
        app.tags.add("zh_cn")


def _remove_non_zh_cn_nodes(app, doctree, docname):
    switcher_version = _resolve_switcher_version(app)
    current_language = getattr(app.config, "language", None)
    is_zh_cn = switcher_version == "zh-cn" or current_language in {"zh_CN", "zh-cn"}
    if is_zh_cn:
        return

    for node in list(
        doctree.findall(
            lambda n: isinstance(n, nodes.Element)
            and "zh-cn-only" in n.get("classes", [])
        )
    ):
        if node.parent is not None and node in node.parent:
            node.parent.remove(node)


def _log_doc_progress(app, docname, source):
    if os.environ.get("READTHEDOCS") == "True":
        print(f"[sphinx] reading: {docname}", flush=True)


def setup(app):
    app.connect("config-inited", _apply_locale_theme_options)
    app.connect("doctree-resolved", _remove_non_zh_cn_nodes)
    app.connect("source-read", _log_doc_progress)
