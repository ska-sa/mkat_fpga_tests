# -*- coding: utf-8 -*-
###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: mmphego@ska.ac.za                                                   #
# Copyright @ 2017 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
import atexit
import glob
import json
import os
import string
import subprocess
import sys
import time

from colors.colors import css_colors

sys.path.append(os.path.abspath('.'))
current_dir = os.path.dirname(os.path.realpath(__file__))

# -- General configuration -----------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.todo',
              'sphinx.ext.ifconfig', 'sphinx.ext.mathjax']
# extensions = ['rst2pdf.pdfbuilder']

[extensions]
todo_include_todos = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
source_encoding = 'utf-8'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'MeerKAT Correlator-Beamformer '
copyright = u'SKA South Africa, 2009-2018'
author = u'Mpho Mphego <mmphego@ska.ac.za>'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
# The short X.Y version.
version = 'Array Release'
# The full version, including alpha/beta/rc tags.
# release = '2'

today_fmt = '%d %B %Y'

# List of documents that shouldn't be included in the build.
#unused_docs = []

# List of directories, relative to source directory, that shouldn't be searched
# for source files.
exclude_trees = []

# The reST default role (used for this markup: `text`) to use for all documents.
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

# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  Major themes that come with
# Sphinx are currently 'default' and 'sphinxdoc'.
html_theme = 'sphinx13'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = ['_themes']

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "CBF Integration Tests"

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_themes/sphinx13/static']

# Folder to exclude when copying html_static_paths
#exclude_dirnames = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_use_modindex = True

# If false, no index is generated.
html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, an OpenSearch de/scription file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# If nonempty, this is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = ''

# Output file base name for HTML help builder.
htmlhelp_basename = 'CBFtestresults'

#---------------------------------------------------------------------------------------------------
#---------------------------------------Options for LaTeX output -----------------------------------
#---------------------------------------------------------------------------------------------------
# Install all Latex package dependencies
# https://docs.typo3.org/typo3cms/extensions/sphinx/AdministratorManual/RenderingPdf/InstallingLaTeXLinux.html

try:
    _json_file = [i for i in glob.iglob('/'.join([current_dir, 'katreport', '*.json'])) if 'latex' in i]
    _json_file = ''.join(_json_file)
    assert os.path.isfile(_json_file)
except Exception:
    pass
else:
    with open(_json_file) as _data:
        _document_data = json.load(_data)
try:
    _document_number, _document_info  = _document_data['document_number'].get(
        _document_data.get('documented_instrument', 'Unknown'), 'Unknown')

    _document_type = ' '.join(['Qualification Test',
        _document_data.get('document_type')[_document_data.get('document_type').keys()[0]]])
    _filename = 'MeerKAT_CBF_%s_%s.tex' % (_document_data.get('document_type').keys()[0],
        time.strftime('%Y%m%d', time.localtime()))
except Exception as e:
    print '%s' % e.message

# http://www.sphinx-doc.org/en/1.4.9/config.html#confval-latex_elements
latex_elements = {
    'papersize': 'a4paper',
    'preamble': r"""
    \geometry{a4paper, total={170mm,257mm}, left=20mm, top=20mm,}
    \usepackage{paralist}
    \usepackage{color}
    % http://users.sdsc.edu/~ssmallen/latex/longtable.html
    \usepackage{longtable}
    \usepackage{amsfonts}
    \usepackage{amssymb}
    % \usepackage{fixltx2e}

    \usepackage{sectsty}
    \sectionfont{\large}
    \subsectionfont{\large}
    \subsubsectionfont{\large}
    \paragraphfont{\large}

    \hypersetup{colorlinks=false,pdfborder={0 0 0},}
    \titleformat{\chapter}[display]
    {\normalfont\bfseries}{}{0pt}{\Large}
    \let\cleardoublepage\clearpage
    """,
    'fontpkg': r'\usepackage{times}',
    'releasename': 'Array Release 3',
    'maketitle': '',
    }


# http://www.sphinx-doc.org/en/1.4.9/config.html#confval-latex_documents
latex_documents = [(
    master_doc,
    '%s' % _filename,
    u'%s %s' %(project, _document_type),
    u'Document Number: %s' % _document_number,
    'manual'),
    ]

#pdf_documents = [('index', u'rst2pdf', u'Sample rst2pdf doc', u'Your Name'),]
# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = 'supplemental/ska-logo.png'

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
latex_use_parts = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_use_modindex = True

# -- Math options --------------------------------------------------------------

# If using pngmath, use preview to correct baseline of rendered math
#pngmath_use_preview = True

# Path to jsMath loader script (relative to ./_static/)
jsmath_path = 'jsMath/easy/load.js'

def replaceAll(_file, searchExp, replaceExp):
    """Search and replace"""
    with open(_file, 'r') as f:
        newlines = []
        for line in f.readlines():
            if searchExp in line:
                newlines.append(line.replace(searchExp, replaceExp))
            else:
                newlines.append(line)

    with open(_file, 'w') as f:
        for line in newlines:
            f.write(line)

def exit_handler():
    """Will execute upon script exit"""
    try:
        _colors = css_colors.keys()
        curpath = os.path.dirname(os.path.realpath(__file__))
        tex_path = '/'.join([curpath, 'build/latex'])
        if os.path.isdir(tex_path):
            tex_file = ''.join(glob.glob('%s/*.tex' % tex_path))
            # Change/ force color change
            for color in _colors:
                replaceAll(tex_file, "DUrole{%s}" % color, "textcolor{%s}" % color)

            old_tags = [
                        # r'\section{',
                        # r"\bigskip\hrule\bigskip",
                        r"{longtable}{|l|l|l|l|l|l|}",
                        r'{} \(',
                        r'$C\)',
                        "\item[",
                        "] \leavevmode",
                        "sphinxstylestrong{",
                        r'\strut}',
                        r'\caption{Summary of Test',
                        r'\caption{Requirements',
                        r'\section{Test Procedure}',
                        r'\(\phi_{TA}(f\) = -2*\pi*f_{RF}*\tau_{TA}\)',
                        r'Fly’92s',
                        r'19‘94 racks',
                        # "{tabulary}{\linewidth}[t]{|T|T|T|T|}",
                        # "{tabulary}{\linewidth}[t]{|T|T|T|}",
                        # "{tabulary}{\linewidth}[t]{|T|T|T|T|T|T|T|T|}",
                        # "end{tabulary}",
                        # "\item{",
                        # "\end{savenotes}",
                        # "\chapter{TP",
                        # "\chapter{AQF",
                        # "\section{Test Configuration}",
                        # "\section{Requirements Verified}",
                        # "\section{Test Procedure}",
                        # r'\sphinxatlongtablestart\begin{longtable}{|l|l|l|l|}',
                        # # r'\\*[\sphinxlongtablecapskipadjust]',
                        # r"\sphinxcaption{Requirements Verification Traceability Matrix}",
                        ]

            new_tags = [
                        # r'\newpage\section{',
                        # r"\vspace{5mm}\bigskip\hrule\bigskip",
                        r"{longtable}[l]{|p{1in}|p{0.6in}|p{1.3in}|p{0.6in}|p{1.3in}|p{0.6in}|}",
                        "{} (",
                        "$C)",
                        "\item\hspace{-0.15cm}",
                        " \leavevmode",
                        "sphinxstylestrong{\small ",
                        r"% \sphinxcaption{Requirements Verification Traceability Matrix}",
                        r'% \caption{Summary of Test',
                        r'% \caption{Requirements',
                        r'\subsection{Test Procedure}',
                        r'$\phi${\footnotesize TA}(f) = -2 * $\pi$ * f{\footnotesize RF} * $\tau{\footnotesize TA}$',
                        r"Fly's",
                        r'19" racks',
                        # "{longtable}[c]{|l|l|l|}",
                        # "{longtable}[c]{|p{1in}|c|c|c|c|c|c|c|}",
                        # "end{longtable}",
                        # "\item",
                        # "\end{savenotes}\\newpage",
                        # "\section{TP",
                        # "\section{AQF",
                        # "\subsection{Test Configuration}",
                        # "\subsection{Requirements Verified}",
                        # "\subsection{Test Procedure}",
                        # r"\sphinxattablestart\centering\sphinxcapstartof{table}",
                        #  '}',
                        # # r'\sphinxaftercaption\begin{longtable}[l]{|p{0.95in}|p{0.6in}|p{1.3in}|p{1.5in}|p{1in}|p{0.6in}|}',
                        ]

            for _old_tags, _new_tags in zip(old_tags, new_tags):
                replaceAll(tex_file, _old_tags, _new_tags)

            docutype = ''.join(_document_data.get('document_type').keys()).lower()
            if docutype == 'qtp':
                _intro_doc = str('/'.join([curpath, 'docs/introduction_%s.tex'%(docutype)]))
            else:
                _intro_doc = str('/'.join([curpath, 'docs/introduction_%s.tex'%(docutype)]))
                old_name = ['DocNumber', 'DocInfo', 'instrument']
                new_name = [_document_number, _document_info, _document_data.get(
                            'documented_instrument', 'Unknown')]

                for _old_name, _new_name in zip(old_name, new_name):
                    replaceAll(_intro_doc, str(_old_name), str(_new_name))
                time.sleep(1)

            replaceAll(tex_file, 'sphinxtableofcontents',
                                 'sphinxtableofcontents\input{%s}' % _intro_doc)
    except Exception:
        pass
    else:
        print '-'*30, 'Done making changes', '-'*30

atexit.register(exit_handler)