# Makefile for Sphinx documentation
#
# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = build

# User-friendly check for sphinx-build
ifeq ($(shell which $(SPHINXBUILD) >/dev/null 2>&1; echo $$?), 1)
$(error The '$(SPHINXBUILD)' command was not found. Make sure you have Sphinx installed, then set the SPHINXBUILD environment variable to point to the full path of the '$(SPHINXBUILD)' executable. Alternatively you can add the directory with the executable to your PATH. If you don't have Sphinx installed, grab it from http://sphinx-doc.org/)
endif

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .
# the i18n builder cannot share the environment and doctrees with the others
I18NSPHINXOPTS  = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source

.PHONY: help clean html latex latexpdf tests1k tests4k tests32k sanitytest

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  latex      to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  latexpdf   to make LaTeX files and run them through pdflatex"
	@echo "  bootstrap  to automagically install Python virtual environment, and all dependencies in .venv"
	@echo "  tests      to run all tests in mkat_fpga_tests/test_cbf.py"
	@echo "  tests1k to run 1k tests in mkat_fpga_tests/test_cbf.py"
	@echo "  tests4k to run 4k tests in mkat_fpga_tests/test_cbf.py"
	@echo "  tests32k to run 32k tests in mkat_fpga_tests/test_cbf.py"
	@echo "  sanitytest to run a sanity tests in mkat_fpga_tests/test_cbf.py"

clean:
	rm -rf $(BUILDDIR)/* || true;
	$(MAKE) clean -C docs/Cover_Page || true;
	rm -rf ".git/index.lock" || true;
	git checkout -- docs/* || true;
	rm -rf -- *.csv *.png *.html || true;
	rm -R ./katreport/*.npy || true;

superclean: clean
	rm -rf -- .venv || true;

html:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

latex:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo
	@echo "Build finished; the LaTeX files are in $(BUILDDIR)/latex."
	@echo "Run \`make' in that directory to run these through (pdf)latex" \
	      "(use \`make latexpdf' here to do that automatically)."

latexpdf:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo "Running LaTeX files through pdflatex..."
	$(MAKE) -C $(BUILDDIR)/latex all-pdf
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."

bootstrap:
	@bash scripts/setup_virtualenv.sh

tests1k:
	@bash -c ". .venv/bin/activate; python run_cbf_tests.py --loglevel=DEBUG --1k"

tests4k:
	@bash -c ". .venv/bin/activate; python run_cbf_tests.py --loglevel=DEBUG --4k"

tests32k:
	@bash -c ". .venv/bin/activate; python run_cbf_tests.py --loglevel=DEBUG --32k"

tests1k_array_release_x:
	@bash -c ". .venv/bin/activate; python run_cbf_tests.py --loglevel=DEBUG --1k --array_release_x"

sanitytest:
	@bash -c ". .venv/bin/activate; echo 'backend: agg' > matplotlibrc; nosetests -sv --logging-level=FATAL --with-xunit --xunit-file=katreport/nosetests.xml --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.test_imaging_data_product_set;"
