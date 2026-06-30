# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXINTL    ?= sphinx-intl
SOURCEDIR     = source
BUILDDIR      = build

# Supported documentation locales (aligned with README language badges).
LOCALES       := zh_CN zh_TW ja ko de fr es it pt_BR

# the i18n builder cannot share the environment and doctrees with the others
I18NSPHINXOPTS  = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) $(SOURCEDIR)
I18NSPHINXLANGS = $(addprefix -l ,$(LOCALES))

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile gettext init-locales sync-readmes translate-po html_i18n \
	html_zh_cn html_zh_tw html_ja html_ko html_de html_fr html_es html_it html_pt_br

init-locales:
	python $(SOURCEDIR)/init_locales.py

sync-readmes:
	python $(SOURCEDIR)/sync_readme_to_po.py

translate-po:
	python $(SOURCEDIR)/translate_po.py --locales ja ko de fr es it pt_BR zh_TW

gettext:
	$(SPHINXBUILD) -b gettext $(I18NSPHINXOPTS) $(BUILDDIR)/locale
	$(SPHINXINTL) update -p $(BUILDDIR)/locale $(I18NSPHINXLANGS)
	python $(SOURCEDIR)/norm_zh.py

html_i18n: html_zh_cn html_zh_tw html_ja html_ko html_de html_fr html_es html_it html_pt_br

html_zh_cn:
	$(SPHINXBUILD) -b html $(SPHINXOPTS) -t zh_cn -D language='zh_CN' "$(SOURCEDIR)" $(BUILDDIR)/html_zh_cn

html_zh_tw:
	$(SPHINXBUILD) -b html $(SPHINXOPTS) -D language='zh_TW' "$(SOURCEDIR)" $(BUILDDIR)/html_zh_tw

html_ja:
	$(SPHINXBUILD) -b html $(SPHINXOPTS) -D language='ja' "$(SOURCEDIR)" $(BUILDDIR)/html_ja

html_ko:
	$(SPHINXBUILD) -b html $(SPHINXOPTS) -D language='ko' "$(SOURCEDIR)" $(BUILDDIR)/html_ko

html_de:
	$(SPHINXBUILD) -b html $(SPHINXOPTS) -D language='de' "$(SOURCEDIR)" $(BUILDDIR)/html_de

html_fr:
	$(SPHINXBUILD) -b html $(SPHINXOPTS) -D language='fr' "$(SOURCEDIR)" $(BUILDDIR)/html_fr

html_es:
	$(SPHINXBUILD) -b html $(SPHINXOPTS) -D language='es' "$(SOURCEDIR)" $(BUILDDIR)/html_es

html_it:
	$(SPHINXBUILD) -b html $(SPHINXOPTS) -D language='it' "$(SOURCEDIR)" $(BUILDDIR)/html_it

html_pt_br:
	$(SPHINXBUILD) -b html $(SPHINXOPTS) -D language='pt_BR' "$(SOURCEDIR)" $(BUILDDIR)/html_pt_br

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
