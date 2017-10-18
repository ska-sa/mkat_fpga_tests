#!/usr/bin/env python
# https://stackoverflow.com/a/44077346
"""
Read a CORE XML backup file and return a JSON file.

Output file is in a format that katreport nose module understands.
Run command with --help to see options.

"""
import xml.etree.ElementTree
import re
import json
import datetime
import hashlib

math_escape_re = re.compile(r'(?P<leading>\s*)\$\$(?P<maths>.+?)\$\$(?P<trailing>\s*)')

def process_latex_escapes(text):
    """Change  $$ latex math $$ text to inline RST math

    Returns (hashed_txt, dict with keys/values {math_hash: rst_math_string})

    hashed_txt : Original text with $$ ... $$ replaced with unique hash strings
    math_hash : Each equation in text is replaced with this string in `hashed_txt`
    rst_math_string : Replace math_hash with this string to get RST after all
                      other RTF processing is done

    """
    replacements = {}
    def replace(m):
        # RST maths barfs if there is leading or trailing whitespace in the
        # :math:`..` role
        math_txt = m.group('maths').strip()
        # Remove RTF garbage (all instances of \RTFGARBAGE, but not \\blah, since \\blah
        # should be unescaped to \blah eventually)
        math_txt = re.sub(r'(?<=[^\\])\\[a-zA-Z0-9]+', '', math_txt)
        # Unescape backslashes and curly braces
        math_txt = math_txt.replace(r'\\', '\\'
            ).replace(r'\{', '{').replace(r'\}', '}')
        math_hash = 'mathhashup' + hashlib.md5(math_txt).hexdigest()
        replacement_txt = '{}:math:`{}`{}'.format(
            m.group('leading'), math_txt, m.group('trailing'))
        leading_space = replacement_txt[0].isspace()
        trailing_space = replacement_txt[-1].isspace()
        # RST maths barfs if there is not leading or trailing whitespace around
        # the :math:`..` role
        if not leading_space:
            replacement_txt = ' ' + replacement_txt
        if not trailing_space:
            replacement_txt = replacement_txt + ' '
        replacements[math_hash] = replacement_txt
        return math_hash

    return math_escape_re.sub(replace, text), replacements

def _clean_xml_tag(input_str):
    """Cleanup the XML tags produced by CORE.

    :return: String. A clean tag.

    """
    if input_str:
        return re.sub(r'{.+}', "", str(input_str)).strip()


def _get_latest_version(attribs):
    """Step through versions and return last text.

    :return: String. Latest text.

    """
    new_text = ''
    for attrib in attribs:
        new_text = attrib.get('text', new_text)

    return new_text


def _build_entity(data):
    """Rearrange the data of a Entity.

    :return: Dict. Rearranged entity.

    """
    # Drop unused terms.
    del_keys = ["attribute-version", "acl", "auditLog", "category"]

    if 'attribute' in data:
        for attrib in data.get('attribute', []):
            if attrib.get('name'):
                if attrib.get('name') not in data:
                    data[attrib.get('name')] = attrib.get('text')
                else:
                    data['attrib-' + attrib.get('name')] = \
                        attrib.get('text')
        del(data['attribute'])

    _list = []
    data['category-list'] = []
    for category in data.get('category', []):
        if category.get('text'):
            _list = category.get('text').split("\\")
            if len(_list) > len(data['category-list']):
                data['category-list'] = _list

    ## UnRTF the text.
    data['description'] = _unrtf(data.get('description', ''))
    rationale = data.get('rationale')
    if rationale:
        data['rationale'] = _unrtf(rationale)

    for key in data:
        if not data.get(key) or data.get(key) == 'nil':
            del_keys.append(key)

    for key in del_keys:
        if key in data:
            del(data[key])
    return data


def _unrtf(text):
    """Remove RTF formating from text.

    :return: String. Text with formating removed.

    """
    # Convert math stuff to hashes
    out_text, maths_dict = process_latex_escapes(text)
    # Convert \par's into newlines
    out_text = re.sub(r'\\par ', '\n\n', out_text)
    # Kill rest of rtf
    out_text = re.sub(r'{\\[\\\w\s;]+;}', '', out_text.strip()).strip()
    out_text = re.sub(r'\\\w+\d*', '', out_text).strip()
    out_text = re.sub(r'\$\$f_\\\{([\w\d]+)\\\}\$\$', r'\1', out_text).strip()

    ## Strip:
    for n in ['{', '}']:
        out_text = out_text.strip(n).strip()

    # Substitute the hashed math thingies back
    for math_hash, math_rst in maths_dict.items():
        out_text = out_text.replace(math_hash, math_rst)

    return out_text


def _update_relationship(base_data, data):
    """Update base_data with relation defined in data."""
    source = data.get("source-id")
    target = data.get("target-id")
    definition = data.get("definition", 'link')
    Xlookup_ref = {'fulfills': 'fulfilled by',
                   'verifies': 'verified by',
                   'X-Employed By': 'Employs',
                   'Built in': 'Built From'}
    Xlookup = {}
    for item in Xlookup_ref:
        value = Xlookup_ref[item]
        Xlookup[item] = value.lower()
        Xlookup[value] = item.lower()
    Xdefinition = Xlookup.get(definition, "X-%s" % definition)

    if source and target:
        # Build the reference.
        if not source in base_data:
            base_data[source] = {definition: []}
        elif not definition in base_data[source]:
            base_data[source][definition] = []
        base_data[source][definition].append(target)

        # Build the cross (X) reference.
        if not target in base_data:
            base_data[target] = {Xdefinition: []}
        elif not Xdefinition in base_data[target]:
            base_data[target][Xdefinition] = []
        base_data[target][Xdefinition].append(source)


def core_xml_to_dict(node, depth=1, path=''):
    """XML to Dict that understands the core format.

    :return: Dict. Core data in a dict.

    """
    depth += 1
    if depth > 9:
        print "Stop",
        return {}

    try:
        data = {'tag': _clean_xml_tag(node.tag),
                'path': path,
                'depth': depth}

        if node.text:
            data['text'] = ''.join(node.xpath('text()')).strip()
        if node.attrib:
            data.update(node.attrib)
        if hasattr(node, 'value'):
            data['value'] = node.value

        for child in node.getchildren():
            try:
                client_tag = _clean_xml_tag(child.tag)
                if client_tag not in data:
                    data[client_tag] = []
                data[client_tag].append(
                    core_xml_to_dict(child, depth, path + "." + client_tag))
            except AttributeError as er:
                print "!", er

        if not data.get('text'):
            data['text'] = _get_latest_version(
                data.get("attribute-version", []))

        return data
    except AttributeError as er:
        print "? ", er, node
        return {'error': str(er)}


def extract_data(base_data, node, depth=1, path=''):
    """Extract the CAM Requirements from the CORE XML.

    Updates the base_data dictionary.

    """
    depth += 1

    try:
        data = {'tag': _clean_xml_tag(node.tag),
                'path': path,
                'depth': depth}

        if node.text:
            data['text'] = ''.join(node.xpath('text()')).strip()
        if node.attrib:
            data.update(node.attrib)
        if hasattr(node, 'value'):
            data['value'] = node.value

        for child in node.getchildren():
            try:
                client_tag = _clean_xml_tag(child.tag)
                if client_tag not in data:
                    data[client_tag] = []
                data[client_tag].append(
                    extract_data(base_data, child,
                                 depth=depth, path=path + "." + client_tag))
            except AttributeError as er:
                print "!", er

        if not data.get('text'):
            data['text'] = _get_latest_version(
                data.get("attribute-version", []))

        if data.get('tag') == 'entity':
            base_data['entity'][data['id']] = _build_entity(data)
            number = base_data['entity'][data['id']].get('number')
            definition = base_data['entity'][data['id']
                                             ].get('definition', '').lower()
            ## To make the logic more readable.
            # Break the masive if statement open.
            del_entry = False
            if 'requirement' in definition:
                if (number and
                   (number.startswith('VR.CM.') or
                        number.startswith('VE.CM.') or
                        number.startswith('R.CM.') or
                        number.startswith('VE.') or
                        number.startswith('TP.C.'))):
                    # If its not a CAM VR or R, drop it.
                    del_entry = False
            elif definition == 'category':
                del_entry = False

            ## Delete the entries we do not want.
            if del_entry:
                del(base_data['entity'][data['id']])
            else:
                # kat_id is a value we intriduce as the value for referencing
                # entities. Normaly the VR number or Timescale name.
                kat_id = None
                if base_data['entity'][data['id']].get('number'):
                    kat_id = base_data['entity'][data['id']].get('number')
                elif base_data['entity'][data['id']].get('name'):
                    kat_id = base_data['entity'][data['id']].get('name')

                if kat_id:
                    base_data['index'][kat_id] = data['id']
                    base_data['entity'][data['id']]['kat_id'] = kat_id

        elif data.get('tag') == 'relationship':
            # Store All the relationships even if we do not have the entities.
            _update_relationship(base_data['relationship'], data)
        elif data.get('tag') == 'core-data':
            base_data['export'] = {}
            for item in ['exported-by', 'time-stamp', 'version']:
                base_data['export'][item] = data.get(item)

        return data
    except AttributeError as er:
        print "? ", er, node
        return {'error': str(er)}


def re_index_to_kat_id(data):
    """Use KAT numbers as the element ID not CORE UUID."""
    new_data = {}
    if not ('relationship' in data and 'entity' in data):
        return {}
    # Create all the entries.
    for num in data.get('index', {}):
        new_data[num] = data['entity'].get(data['index'][num])
        new_data[num]['relationship'] = {}

    # Build up relationships and X-relationships.
    for num in data.get('index', {}):
        relationship = data['relationship'].get(new_data[num]['id'], {})
        for rel_type in relationship:
            if rel_type not in new_data[num]['relationship']:
                new_data[num]['relationship'][rel_type] = []
            for rel_id in relationship[rel_type]:
                rel_num = data['entity'].get(rel_id, {}).get('kat_id')
                if rel_num:
                    new_data[num]['relationship'][rel_type].append(rel_num)
                    if (rel_type == 'categorized by'
                            and rel_num.startswith("Timescale")):
                        new_data[num]['timescale'] = rel_num
            if not new_data[num]['relationship'][rel_type]:
                del(new_data[num]['relationship'][rel_type])

    # Update timescale.
    for num in [n for n in new_data
                if new_data[n]["definition"] == "VerificationRequirement"]:
        timescale = None
        for req in new_data[num]['relationship'].get('verifies', []):
            timescale = new_data[req].get('timescale', timescale)
        if timescale:
            new_data[num]['timescale'] = timescale

    if not '__Meta' in new_data:
        new_data['__Meta'] = {}
    if data.get('export'):
        new_data['__Meta']['export'] = data.get('export')
        new_data['__Meta']['processed'] = \
            {'time': str(datetime.datetime.utcnow())}
    return new_data


def process_xml_to_json(xml_filename, json_filename,
                        no_filter=False, verbose=False, log_func=None):
    """Process an CORE XML file and output a JSON file.

    :param xml_filename: String. Filename of Core XML backup.
    :param json_filename: String. Filename of output JSON file.
    :param no_filter: Boolea.: False - Filter the JSON output.
    :param verbose: Boolean. True be verbose.

    """

    if log_func is None:
        def _redef_log_func(*msg):
            print " ".join([str(s) for s in msg])
        log_func = _redef_log_func

    if verbose:
        log_func("Parse File:", xml_filename)
    try:
        import lxml.etree
        parser = lxml.etree.XMLParser(recover=True)
        doc = xml.etree.ElementTree.parse(xml_filename, parser)
    except ImportError:
        print "Install Python lxml for better processing of XML files."
        doc = xml.etree.ElementTree.parse(xml_filename)

    if verbose:
        log_func("Step through CORE data and create output.")
    base = {'entity': {}, 'relationship': {}, 'index': {}}

    if no_filter:
        log_func("Only do XML to JSON conversion")
        ri_data = core_xml_to_dict(doc.getroot())
    else:
        extract_data(base, doc.getroot())
        if verbose:
            log_func("ReIndex data.")

        with open(json_filename + ".test.json", 'w') as fh:
            fh.write(json.dumps(base, indent=4))

        ri_data = re_index_to_kat_id(base)

    if verbose:
        log_func("Write to file ", json_filename)
    with open(json_filename, 'w') as fh:
        fh.write(json.dumps(ri_data, indent=4))

if __name__ == '__main__':
    import os
    import sys
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-i", "--input_file", dest="in_filename",
                      help="name of CORE XML file to read.", metavar="FILE")
    parser.add_option("-o", "--output_file", dest="out_filename",
                      help="name of file to write to.", metavar="FILE")
    parser.add_option("-u", "--unfiltered", dest="no_filter",
                      action="store_true",
                      help="Remove filtering and processing for katreport"
                           "only convert the XML to JSON")
    parser.add_option("-v", "--verbose", dest="verbose", action="store_true",
                      help="name of file to write to.", metavar="FILE")

    (options, args) = parser.parse_args()
    if not (options.in_filename and os.path.isfile(options.in_filename)):
        print "Input file must be given and exists."
        sys.exit(1)
    if not options.out_filename:
        print "No output file given."
        sys.exit(2)

    process_xml_to_json(options.in_filename, options.out_filename,
                        no_filter=options.no_filter,
                        verbose=options.verbose)
