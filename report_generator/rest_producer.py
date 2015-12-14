###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
"""Helper routines for programtically producing ReStructuredText output."""


class ReStProducer(object):

    """A very minimal, very stupid ReStructuredText producer."""

    # Each value is a tuple with the character to use for headings of
    # various levels and True if an overline should be added
    heading_levels = dict(
        part=('#', True),
        chapter=('*', True),
        section=('=', False),
        subsection=('-', False),
        subsubsection=('^', False),
        paragraph=('"', False))

    indent = ' ' * 4

    def __init__(self):
        self._header = None
        self._output = None
        self.clear()

    def _ensure_empty_line(self):
        try:
            prevline = self._output[-1]
        except IndexError:
            prevline = ''

        # Add a blank line if one the last line is not blank already
        if prevline:
            self._output.append('')

    def add_heading(self, level, heading, anchor=False):
        """Add a headin.

        :param level: Str. Type of heading.
        :param heading: Str. Title of the heading.
        :param anchor: Boolean. If a anchor should be added, the anchor will
                        allow the heading to be referenced.

        """
        assert('\n' not in heading)
        if anchor:
            self.add_anchor(heading)
        self._ensure_empty_line()
        headchar, overline = self.heading_levels[level.lower()]
        if overline:
            self._output.append(headchar * len(heading))
        self._output.append(heading)
        self._output.append(headchar * len(heading))
        self._output.append('')

    def add_sourcecode(self, quote):
        """Add sourceode block, i.e. Quote text verbatim in a block-quote."""
        if quote:
            self._ensure_empty_line()
            quote = str(quote).replace('\r', '\n').split('\n')
            self._output.append('')
            self._output.append('::')
            self._output.append('')
            for q in quote:
                self._output.append(self.indent + q.strip())
            self._output.append('')
            self._output.append('..')
            self._output.append('')

    def add_figure(self, filename, caption, alt=None, legend=None,
                   align='center', figwidth='90%', width='80%', **options):
        self._ensure_empty_line()
        self._output.append(' .. figure:: {}'.format(filename))
        if alt:
            self._output.append(self.indent + ':alt: {}'.format(alt))
        if align:
            self._output.append(self.indent + ':align: {}'.format(align))
        if figwidth:
            self._output.append(self.indent + ':figwidth: {}'.format(figwidth))
        if width:
            self._output.append(self.indent + ':width: {}'.format(width))
        for option, value in options.items():
            if value:
                self.output.append(self.indent + ':{}: {}'.format(option, value))
        self._ensure_empty_line()
        self.add_indented_raw_text(caption, level=1)
        if legend:
            self._ensure_empty_line()
            self.add_indented_raw_text(legend, level=1)
        self._ensure_empty_line()

    def add_raw_text(self, text):
        """Add raw text to the ReST output stream."""
        if text:
            self._output.append(str(text))

    def add_indented_raw_text(self, text, level=1):
        for line in str(text).split('\n'):
            self._output.append(self.indent*level + line)

    def add_line(self, text):
        if text:
            self._output.append(text)
            self._output.append('')

    def add_paragraph(self, text):
        if text:
            self._ensure_empty_line()
            self._output.append(text)
            self._output.append('')

    def _unslug_str(self, label):
        """The oposite of slugifying a string."""
        return label.strip().lower().replace("_", "-").replace(".", "-")

    def add_anchor(self, label):
        if label:
            self._ensure_empty_line()
            self._output.append(".. _%s:" % self._unslug_str(label))
            self._output.append('')

    def add_link(self, label, title=None):
        """Insert a reference to an anchor."""
        if title:
            return ":ref:`%s <%s>`" % (title, self._unslug_str(label))
        else:
            return ":ref:`%s`" % self._unslug_str(label)

    def add_table(self, table_header, table_data, table_title=None):
        """Draw a simple table."""
        if table_data:
            self._ensure_empty_line()
            if not table_title:
                table_title = "Table:"
            self._output.append(".. csv-table:: %s" % table_title)
            self._output.append("   :header: %s" %
                                ",".join(['"%s"' % x for x in table_header]))
            self._output.append('')
            for row in table_data:
                self._output.append("   %s" %
                                    ",".join(['"%s"' % x for x in row]))
            self._output.append('')

    def add_table_ld(self, data, table_title=None, header_map=None,
                     hide_first_header=False):
        """Given a list of dictionaries [{}, {}, {}] draw a table."""
        if not header_map:
            # Will use header map to change the labels of columns and the order
            # of the coloumns.
            header_map = {}
        if not table_title:
            table_title = "Table:"

        table_header = set()
        if data:
            for line in data:
                for key in line.keys():
                    table_header.add(key)
            table_header = sorted(table_header)
            if header_map:
                table_header = header_map
            self._ensure_empty_line()
            self._output.append(".. csv-table:: %s" % table_title)
            header = ['"%s"' % x.title() for x in table_header]
            if hide_first_header:
                header[0] = " "
            self._output.append("   :header: %s" % ",".join(header))
            self._output.append('')
            for line in data:
                self._output.append("   %s" %
                                    ",".join(['"%s"' % line.get(x, '--')
                                              for x in table_header]))
            self._output.append('')

    def add_table_dd(self, data, table_title=None, header_map=None):
        """Given a dict of dictionaries {'a': {}, 'b': {}} draw a table."""
        if not header_map:
            # Will use header map to change the labels of columns and the order
            # of the coloumns.
            header_map = {}
        if not table_title:
            table_title = "Table:"

        table_header = set()
        if data:
            for line in data:
                for key in data[line].keys():
                    table_header.add(key)
            self._ensure_empty_line()
            self._output.append("BALBLA %s" % str(table_header))

            self._ensure_empty_line()
            #TODO(Martin): Do some magic with header map to get the order.
            table_header = sorted(table_header)

            self._ensure_empty_line()
            self._output.append(".. csv-table:: %s" % table_title)
            self._output.append("   :header: \"zzz\",%s" %
                                ",".join(['"%s"' % x for x in table_header]))
            self._output.append('')
            for line in data.keys():
                self._output.append("  \"%s\",%s" % (line,
                                    ",".join(['"%s"' % data[line].get(x, '--')
                                              for x in table_header])))
            self._output.append('')

    def add_box_see_also(self, text):
        """Add a see also box to the document."""
        if text:
            self._ensure_empty_line()
            self._output.append(".. seealso:: %s" % text)
            self._output.append('')

    def add_box_todo(self, text):
        """Add a todo box to the document."""
        if text:
            self._ensure_empty_line()
            self._output.append(".. todo:: %s" % text)
            self._output.append('')

    def add_box_note(self, text):
        """Add a note box to the document."""
        if text:
            self._ensure_empty_line()
            self._output.append(".. note:: %s" % text)
            self._output.append('')

    def add_box_warning(self, text):
        """Add a warning box to the document."""
        if text:
            self._ensure_empty_line()
            self._output.append(".. warning:: %s" % text)
            self._output.append('')

    def add_include(self, filename):
        if filename:
            self._ensure_empty_line()
            self._output.append(".. include:: %s" % filename)
            self._output.append('')

    def clear(self):
        """Clear the current document."""
        self._output = []
        self._header = set()

    def clean_text_block(self, text):
        """Atempt to cleanup a text block to be better displayed in RST.

        :param text: String,
        :return: String.

        """
        new_text = []
        prev = 100000
        for line in text.splitlines():
            indent = len(line) - len(line.lstrip())
            if indent > prev:
                new_text.append(' ')
            prev = indent
            new_text.append(line)
        return "\n".join(new_text)

    def str_style(self, style, text=None):
        """Add style to the heading and output this string with style.

        :return: String. Text with style.

        """
        # Standard HTML colours. (from w3)
        # aqua, black, blue, fuchsia, gray, green, lime, maroon, navy,
        # olive, orange, purple, red, silver, teal, white, yellow
        if not text:
            text = style
        text = str(text)
        ## Predefined Styles.
        if style.lower() == 'bold':
            return "**%s**" % text
        elif style.lower() == 'italics':
            return "*%s*" % text
        else:
            style = {'error': 'orange',
                     'fail': 'red',
                     'failed': 'red',
                     'skip': 'blue',
                     'skipped': 'blue',
                     'tbd': 'blue',
                     'pass': 'green',
                     'passed': 'green',
                     'exists': 'green',
                     'waived': 'fuchsia',
                     'control': 'gray',
                     'checkbox': 'lime',
                     'not implemented': 'orange',
                     'unknown': 'gray',
                     }.get(style.lower())
            if style:
                style_line = ".. role:: %s" % style
                self._header.add(style_line)
                return ":%s:`%s`" % (style, text)
            else:
                return text

    def write(self, fh):
        """Write the oupyt to a file."""
        if isinstance(fh, file):
            for line in self.output:
                fh.write(line + '\n')
        else:
            with open(fh, 'w') as filehandle:
                for line in self.output:
                    filehandle.write(line + "\n")

    @property
    def output(self):
        """The RST data."""
        for line in self._header:
            yield line
            yield ''
        for line in self._output:
            yield line
#
