# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXINTL    ?= sphinx-intl
SOURCEDIR     = source
BUILDDIR      = build

# the i18n builder cannot share the environment and doctrees with the others
I18NSPHINXOPTS  = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) $(SOURCEDIR)
I18NSPHINXLANGS = -l zh_CN

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile gettext build-mo html_i18n html_zh_cn html_de

# Usage:
#   make html_i18n LOCALE=de
#   make html_de
#   make html_zh_cn
html_i18n:
	python build_i18n.py $(LOCALE)
	$(SPHINXBUILD) -b html $(SPHINXOPTS) $(if $(filter zh_CN,$(LOCALE)),-t zh_cn,) -D language='$(LOCALE)' "$(SOURCEDIR)" $(BUILDDIR)/html_$(LOCALE)

html_de:
	$(MAKE) html_i18n LOCALE=de

html_zh_cn:
	$(MAKE) html_i18n LOCALE=zh_CN

build-mo:
	python build_i18n.py --all

gettext:
	$(SPHINXBUILD) -b gettext $(I18NSPHINXOPTS) $(BUILDDIR)/locale
	$(SPHINXINTL) update -p $(BUILDDIR)/locale $(I18NSPHINXLANGS)
	python $(SOURCEDIR)/norm_zh.py

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)