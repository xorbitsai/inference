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
I18NSPHINXLANGS = -l zh_CN -l ja_JP

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile html_zh_cn html_ja_jp gettext

html_zh_cn:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) -t zh_cn -D language='zh_CN' "$(SOURCEDIR)" $(BUILDDIR)/html_zh_cn

html_ja_jp:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) -t ja_jp -D language='ja_JP' "$(SOURCEDIR)" $(BUILDDIR)/html_ja_jp

gettext:
	$(SPHINXBUILD) -b gettext $(I18NSPHINXOPTS) $(BUILDDIR)/locale
	$(SPHINXINTL) update -p $(BUILDDIR)/locale $(I18NSPHINXLANGS)
	python $(SOURCEDIR)/norm_zh.py

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)