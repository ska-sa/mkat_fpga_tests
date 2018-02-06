#!/usr/bin/env python
# https://stackoverflow.com/a/44077346
###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cbf@ska.ac.za                                                       #
# Maintainer: mmphego@ska.ac.za, alec@ska.ac.za                               #
# Copyright @ 2016 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################

import coloredlogs
import glob
import json
import katcp
import os
import platform
import pwd
import py_compile
import re
import report_generator
import sh
import subprocess
import sys
import threading
import time

from optparse import OptionParser
from process_core_xml import process_xml_to_json
from PyPDF2 import PdfFileMerger
from report_generator.report import Report
from shutil import copyfile
from signal import SIGKILL


# List all core test python module dependencies
_core_dependencies = ['corr2', 'casperfpga', 'spead2', 'katcp']

# TODO (MM) perhaps implement a config file
_revision ='1.0'
_release = '2/3'

def option_parser():
    usage = """
        Usage: %prog [options]
        This script auto executes CBF Tests with selected arguments.
        See Help for more information.
        """
    parser = OptionParser(usage)
    parser.add_option('--loglevel',
                      action="store",
                      default="INFO",
                      dest='log_level',
                      help='log level to use, default INFO, options INFO, DEBUG, WARNING, ERROR')

    parser.add_option("-q", "--quiet",
                      action="store_false",
                      dest="verbose",
                      help="Be more quiet")

    parser.add_option("--nose",
                      dest="nose_args",
                      action="store",
                      type="string",
                      default=None,
                      help="""Additional arguments to pass on to nosetests.
                      eg: --nosetests \"-x -s -v\"""")

    parser.add_option("--acceptance",
                      dest="site_acceptance",
                      action="store_true",
                      default=False,
                      help="Will only run test marked '@site_acceptance' or "
                           " if in the Karoo(site) then also @site_only tests")

    parser.add_option("--instrument-activate",
                      dest="instrument_activate",
                      action="store_true",
                      default=False,
                      help=("""launch an instrument. eg:
                            ./run_cbf_tests.py -v --instrument-activate --4A4k"""))

    parser.add_option("--dry_run",
                      dest="dry_run",
                      action="store_true",
                      default=False,
                      help="Do a dry run. Print commands that would be called as well as generate"
                           "test procedures")

    parser.add_option("--available-tests",
                      dest="available-tests",
                      action="store_true",
                      default=False,
                      help="Do a dry run. Print all tests available")

    parser.add_option("--4A4k",
                      action="store_const",
                      const='bc8n856M4k',
                      dest="mode",
                      default=None,
                      help="Run the tests decorated with @instrument_bc8n856M4k")

    parser.add_option("--4A32k",
                      action="store_const",
                      const='bc8n856M32k',
                      dest="mode",
                      default=None,
                      help="Run the tests decorated with @instrument_bc8n856M32k")

    parser.add_option("--8A4k",
                      action="store_const",
                      const='bc16n856M4k',
                      dest="mode",
                      default=None,
                      help="Run the tests decorated with @instrument_bc16n856M4")

    parser.add_option("--8A32k",
                      action="store_const",
                      const='bc16n856M32k',
                      dest="mode",
                      default=None,
                      help="Run the tests decorated with @instrument_bc16n856M32k")

    parser.add_option("--16A4k",
                      action="store_const",
                      const='bc32n856M4k',
                      dest="mode",
                      default=None,
                      help="Run the tests decorated with @instrument_bc32n856M4k")

    parser.add_option("--16A32k",
                      action="store_const",
                      const='bc32n856M32k',
                      dest="mode",
                      default=None,
                      help="Run the tests decorated with @instrument_bc32n856M32k")

    parser.add_option("--quick",
                      dest="katreport_quick",
                      action="store_true",
                      default=False,
                      help="Only generate a small subset of the reports")

    parser.add_option("--with_html",
                      dest="gen_html",
                      action="store_true",
                      default=False,
                      help="Generate HTML report output")

    parser.add_option("--QTP",
                      dest="gen_qtp",
                      action="store_true",
                      default=False,
                      help="Generate PDF report output with Qualification Test Procedure")

    parser.add_option("--QTR",
                      dest="gen_qtr",
                      action="store_true",
                      default=False,
                      help="Generate PDF report output with Qualification Test Results")

    parser.add_option("--no_slow",
                      dest="slow_test",
                      action="store_false",
                      default=True,
                      help="Exclude tests decorated with @slow in this test run")

    parser.add_option("--report",
                      dest="report",
                      action="store",
                      type="string",
                      default='local_&_test',
                      help="Only generate the reports. No tests will be run.\n"
                      "Valid options are: local, jenkins, skip and results. "
                      "'results' will print the katreport[_accept].json test results")

    parser.add_option("--clean",
                      dest="cleanup",
                      action="store_true",
                      default=False,
                      help="""Cleanup reports from previous test run. Reports
                      are replaced by default without --clean. Clean is
                      useful with --quick to only generate the html of the
                      test run report""")

    parser.add_option("--dev_update",
                      dest="dev_update",
                      action="store_true",
                      default=False,
                      help="Do pip install update and install latest packages")

    # parser.add_option("--jenkins",
    #                   dest="jenkins",
    #                   action="store_true",
    #                   default=False,
    #                   help="Run this command with the correct flags for jenkins.")

    # parser.add_option("--manual_systype",
    #                   dest="manual_systype",
    #                   action="store",
    #                   default=None,
    #                   help="Overwrite the system systype used for report generation on jenkins.")

    (cmd_options, cmd_args) = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)
    return cmd_options, cmd_args

def run_command(settings, cmd, log_filename=None, stdout=False, stderr=False, shell=False):
    if log_filename is False:
        log_filename = '/dev/null'

    if settings.get('dry_run') or shell:
        # Fundge the command to add " when the command is
        # printed or run as string in shell.
        # if sh.which('nosetests') in cmd:
        if '/usr/local/bin/nosetests' in cmd:
            for item in range(len(cmd)):
                if cmd[item].startswith("-A"):
                    # break
                    cmd[item] = cmd[item].replace('-A(', '-A"(') + '"'

    if settings.get('dry_run'):
        # Just print the command
        os.environ['DRY_RUN'] = 'True'
        # settings['gen_qtp'] = True

    if settings.get('available-tests'):
        # nosetests -vv collect-only
        if cmd[0].endswith('nosetests'):
            cmd.insert(1, '--collect-only')
            logger.info('%s' % cmd)

    if log_filename and not stdout and not stderr:
        with open(log_filename, 'w') as fh:
            logger.debug('Writing %s to file: %s' % (' '.join(cmd), log_filename))
            return subprocess.call(cmd, stderr=fh, stdout=fh)
    else:
        logger.debug('Run command with stderr: %s' %cmd)
        kwargs_cmd = {'env': os.environ}
        try:
            if shell:
                kwargs_cmd['shell'] = True
                str_cmd = " ".join(cmd)
                logger.debug('Run command with shell: %s' % str_cmd)
                return subprocess.call(str_cmd, **kwargs_cmd)
            else:
                return subprocess.call(cmd, **kwargs_cmd)
        except KeyboardInterrupt as e:
            kill_pid('nosetests')
            msg = "Test closed prematurely, and process has since been killed"
            logger.exception(msg)
            raise RuntimeError(msg)


def run_command_output(settings, cmd):
    """Run a command and return output."""
    if settings.get('dry_run') and 'nosetests' in cmd:
        logger.info('Dry-Run with command: %s' %cmd)
        return
    else:
        try:
            return subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=open('/dev/null', 'a')).communicate()[0]
        except OSError:
            logger.error('Failed to execute command: %s' % cmd)
            return


class RunCmdTimeout(threading.Thread):
    def __init__(self, settings, cmd, timeout):
        """
        Run a command with timeout
        """
        threading.Thread.__init__(self)
        self.cmd = cmd
        self.timeout = timeout

    def run(self):
        if settings.get('dry_run'):
            logger.info('Dry running: %s' % cmd)
            return
        else:
            try:
                if settings.get('log_level') == 'DEBUG':
                    self.p = subprocess.Popen(self.cmd)
                else:
                    self.p = subprocess.Popen(self.cmd, stdout=open('/dev/null', 'a'),
                        stderr=subprocess.PIPE)
                self.p.wait()
            except OSError:
                logger.exception('Failed to execute command')
                return

    def run_the_process(self):
        try:
            self.start()
            self.join(self.timeout)
        except KeyboardInterrupt:
            logger.exception('Failed to execute command')
        if self.is_alive():
            self.p.terminate()
            self.join()
            return self.p.returncode
        else:
            try:
                return self.p.returncode
            except Exception:
                logger.exception('Failed to execute command')
                return


def do_dev_update(settings):
    """Do a code update and install."""
    logger.debug("Performing a Python package upgrade")
    os.chdir(settings.get('me_dir'))
    _pip = sh.which('pip')
    run_command(settings, [_pip, 'install', '-U', 'pip'])
    run_command(settings, [_pip, 'install', '-U', '-r', 'pip-requirements.txt'])

def get_system_info():
    _system = os.uname()
    system = {'system_type': '', 'site': '', 'system_config': ''}
    # system = {'systype': '', 'site': '', 'nodetype': 'jenkins_build_slave'}
    system['sysname'] = os.uname()[1]
    # Backwards compatability.
    if _system[1].startswith('cmc'):
        system['site'] = True
        system['system_location'] = 'SKA SA - Karoo'
    elif _system[1].startswith('dbe'):
        system['site'] = False
        system['system_location'] = 'SKA SA - CBF Lab'
    else:
        # Unknown system location, and assuming it is not on site
        system['site'] = False
        system['system_location'] = None
    # system['system_type'] = system.get('systype')
    logger.debug('Retrieved system information')
    return system

def process_core_data(settings):
    """Process the CORE XML file if JSON file is not there or old."""

    # Setup.
    if settings.has_key('me_dir'):
        temp_core_export = '/'.join([settings['me_dir'], 'tmp', 'CORE_EXPORT'])
        if not os.path.exists(temp_core_export):
            os.makedirs(temp_core_export)

    if 'tmp_core_dir' not in settings:
        settings['tmp_core_dir'] = temp_core_export
    if 'xml_file' not in settings:
        settings['xml_file'] = os.path.join(settings['tmp_core_dir'], "svn/MeerKAT.xml")
    if 'json_file' not in settings:
        settings['json_file'] = os.path.join(settings['tmp_core_dir'], "M.json")
    settings['xml_mtime'] = 0
    settings['json_mtime'] = 0
    settings['use_core_json'] = False
    logger.debug('Processing CORE Data')
    # Update SVN.
    if not os.path.isdir(settings['tmp_core_dir']):
        os.mkdir(settings['tmp_core_dir'])
    os.chdir(settings['tmp_core_dir'])
    try:
        core_supplemental_dir = os.path.join(settings['me_dir'], 'supplemental', "MeerKAT.xml")
        settings['xml_file'] = core_supplemental_dir
        assert os.path.exists(core_supplemental_dir)
    except AssertionError as e:
        logger.warning('CORE.xml does not exist in directory: %s' % core_supplemental_dir)
        core_backup = settings.get('core_backup')
        logger.debug('Retrieving CORE.xml from backup dir: %s' % core_backup)
        if os.path.exists(core_backup):
            latest_core_xml = max(glob.iglob(os.path.join(core_backup ,'*.[Xx][Mm][Ll]')),
                key=os.path.getctime)
            logger.debug('CORE.xml file name %s @ %.2f Mb' % (os.path.split(latest_core_xml)[-1]
              , os.path.getsize(latest_core_xml) / 1e6))
            settings['xml_file'] = latest_core_xml
        else:
            errmsg = 'CORE.xml file does not exist in %s dir' % core_backup
            logger.error(errmsg)
            raise RuntimeError(errmsg)

    if os.path.isfile(settings['xml_file']):
        settings['xml_mtime'] = os.path.getmtime(settings['xml_file'])

    if os.path.isfile(settings['json_file']):
        settings['json_mtime'] = os.path.getmtime(settings['json_file'])

    # Update JSON
    if (settings['xml_mtime'] > 0 and
            settings['json_mtime'] < settings['xml_mtime']):
        logger.debug("Process: XML -> JSON")
        # if not settings.get('dry_run'):
        process_xml_to_json(settings['xml_file'], settings['json_file'], verbose=True)
    else:
        logger.debug("JSON File is up-to-date")
    if os.path.isfile(settings['json_file']):
        settings['use_core_json'] = True

def create_log_func(settings):
    """Return the log function.

    :param settings: Dict. The settings for this program.

    :return: func.

    """
    def __log_func(level=None, *args):
        if level and not args:
            args = [level]
            level = 'DEBUG'
        else:
            level = level.upper()

        if level not in log_levels:
            args = [level] + list(args)
            level = 'DEBUG'
        # Todo(Martin): Filter verbosity on message level.
        if level in allowed_levels:  # Closure
            print "%s"%(time.strftime("%H:%M:%S", time.localtime())), level, "++", " ".join(
                [str(s) for s in args])

    log_levels = ['ERROR', 'WARNING', 'INFO', 'DEBUG']
    allowed_levels = ['ERROR', 'WARNING', 'INFO']
    if settings['verbose'] is True:
        allowed_levels.append('DEBUG')
    elif settings['verbose'] is False:
        allowed_levels.pop()
        allowed_levels.pop()
    return __log_func

def replaceAll(file, searchExp, replaceExp):
    """Search and replace"""
    with open(file, 'r') as f:
        newlines = []
        for line in f.readlines():
            newlines.append(line.replace(searchExp, replaceExp))
    with open(file, 'w') as f:
        for line in newlines:
            f.write(line)

def generate_sphinx_docs(settings):
    os.chdir(settings['base_dir'])
    base_dir = str(settings.get('base_dir', os.path.dirname(os.path.realpath(__file__))))
    katreport_dir = settings.get('katreport_dir', 'katreport_dir')
    katreport_path = os.path.join(base_dir, katreport_dir)
    log_file = '/dev/null'

    # QTP Index data file
    _Index_QTP = """
    .. toctree::
       :maxdepth: 5
       :hidden:

    .. _kattests:

    .. _conventions: http://sphinx.pocoo.org/rest.html

    .. role:: red
    .. role:: darkred
    .. role:: fuchsia
    .. role:: orange
    .. role:: blue
    .. role:: green
    .. role:: yellow


    .. toctree::
       :glob:

       katreport/cbf_timescale_unlinked_qualification_procedure.rst
    """
    # ------------------------------------------------------
    # QTR Index data file
    _Index_QTR = """
    .. toctree::
       :maxdepth: 5
       :hidden:


    .. _kattests:

    .. _conventions: http://sphinx.pocoo.org/rest.html

    .. role:: red
    .. role:: darkred
    .. role:: fuchsia
    .. role:: orange
    .. role:: blue
    .. role:: green
    .. role:: yellow

    .. toctree::
       :glob:

       katreport/cbf_timescale_unlinked_qualification_results.rst
       katreport/katreport_system.rst
    """
    def verbose_cmd_exec(log_level, cmd):
        if settings.get('log_level') == log_level:
            logger.debug('Executed CMD: %s' % cmd)
            status = run_command(settings, cmd)
        else:
            status = run_command(settings, cmd, log_file)

    try:
        logger.debug(
          'Cleaning up previous builds, Note: Backup can be found on ../CBF_Tests_Reports dir')
        verbose_cmd_exec('DEBUG', ['make', 'clean'])
        assert settings.get('gen_html', False)
    except AssertionError:
        pass
    else:
        logger.info("Generating HTML document from rst files")
        verbose_cmd_exec('DEBUG', ['make', 'html'])

    try:
        assert settings.get('gen_qtp', False) or settings.get('gen_qtr', False)
    except AssertionError:
        logger.warning("Note: NO LATEX/PDF DOCUMENT WILL BE CREATED")
        return
    else:
        logger.info("Generating LATEX/PDF document from reST")
        document_data = {
                'project': 'MeerKAT Correlator-Beamformer Array Release 2/3 Qualification Test ',
                'documented_instrument': settings.get('system_type', 'Unknown'),
                'document_number': {'QTP': 'M1200-0000-054 ',
                                    'QTR': 'M1200-0000-055',
                                    'bc8n856M4k': ['M1200-0000-055-1',
                                                   '4 Antenna System running in Wideband Coarse (4K) mode'
                                                   ' with a beamformer'],
                                    'bc8n856M32k':  ['M1200-0000-055-2',
                                                   '4 Antenna System running in Wideband Fine (32K) mode'],
                                    'bc16n856M4k':  ['M1200-0000-055-3',
                                                   '8 Antenna System running in Wideband Coarse (4K) mode'
                                                   ' with a beamformer'],
                                    'bc16n856M32k': ['M1200-0000-055-4',
                                                   '8 Antenna System running in Wideband Fine (32K) mode'],
                                    'bc32n856M4k':  ['M1200-0000-55-5',
                                                   '16 Antenna System running in Wideband Coarse (4K) mode'
                                                   ' with a beamformer'],
                                    'bc32n856M32k': ['M1200-0000-55-6',
                                                   '16 Antenna System running in Wideband Fine (32K) mode'],
                                    'bc64n856M4k':  ['M1200-0000-55-7',
                                                   '32 Antenna System running in Wideband Coarse (4K) mode'
                                                   ' with a beamformer'],
                                    'bc64n856M32k': ['M1200-0000-55-8',
                                                   '32 Antenna System running in Wideband Fine (32K) mode'],
                                    'bc128n856M4k':  ['M1200-0000-55-9',
                                                   '64 Antenna System running in Wideband Coarse (4K) mode'
                                                   ' with a beamformer'],
                                    'bc128n856M32k': ['M1200-0000-55-10',
                                                   '64 Antenna System running in Wideband Fine (32K) mode'],
                                    },
                        }
        _filename = os.path.join(settings.get('me_dir'),
            settings.get("katreport_dir"), 'latex_data.json')
        if settings.get('gen_qtp'):
            logger.info("Generating Qualification Test Procedure (PDF) document.")
            document_data['document_type'] = {'QTP': 'Procedure'}
            generate_index(_Index_QTP)
        elif settings.get('gen_qtr'):
            logger.info("Generating Qualification Test Results (PDF) document.")
            document_data['document_type'] = {'QTR': 'Results'}
            generate_index(_Index_QTR)
        else:
            logger.error('You need to specify which document you need to produce')
            return
        with open(_filename, 'w') as fh:
            fh.write(json.dumps(document_data, indent=4))
        logger.debug('Let me sleep for a while')
        time.sleep(2)
        cmd = ['make', 'latexpdf']
        if settings.get('log_level') == 'DEBUG':
            status = run_command(settings, cmd)
        else:
            logger.info('Note: Generating documents quietly.')
            status = run_command(settings, cmd, log_file)

    if status:
        logger.error("Error occurred while making document, bailing!!")
        return
    else:
        latex_dir = str(os.path.abspath('/'.join([base_dir, 'build/latex'])))
        docs_dir = (os.path.abspath(os.path.join(base_dir, 'docs')))
        cover_page_dir = str(os.path.abspath(os.path.join(docs_dir, 'Cover_Page/')))
        latex_pdf = max(glob.iglob(latex_dir + '/*.pdf'), key=os.path.getctime)
        if os.path.exists(latex_dir) and os.path.exists(cover_page_dir) and os.path.exists(latex_pdf):
            logger.debug("Generating MeerKAT cover page")
            # ToDo: MM 16-Nov-2017 Improve this logic
            _document_title = document_data['project'] + document_data['document_type'].values()[0]
            _document_type = '%s: Qualification Test %s' % (document_data['document_type'].keys()[0],
                document_data['document_type'].values()[0])
            _document_num = document_data.get('document_number', 'Unknown').get(
                document_data.get('document_type', 'Unknown').keys()[0])
            _document_rel = _revision
            latex_file = max(glob.iglob(cover_page_dir + '/*.tex'), key=os.path.getctime)
            orig_names = ['Doctype', 'DocNumber', 'DocRevision', 'DocumentTitle']
            if settings.get('gen_qtr', False):
                logger.debug('Making fixes for QTR on Cover page')
                num_antennas = ''.join(['(',document_data['document_number'].get(
                    document_data.get('documented_instrument', 'Unknown'), 'Unknown')[-1].split(
                    'Antenna')[0],  'A Fully Tested) '])
                _document_num = document_data['document_number'].get(
                    document_data.get('documented_instrument', 'Unknown'), 'Unknown')[0]
                _document_title = _document_title.replace('Qualification',
                    num_antennas + 'Qualification')
                # TODO (MM) Find a way not to hardcode this info
                replaceAll(latex_file, '{Performed by}{}{}',
                    '{Performed by}{M. Mphego}{Test \& Verification Engineer}')

            new_names = [_document_type, _document_num, _document_rel, _document_title]
            for _new, _old in zip(new_names, orig_names):
                replaceAll(latex_file, _old, _new)
            cmd = ['make', 'clean', '-C', cover_page_dir]
            if settings.get('log_level') == 'DEBUG':
                status = run_command(settings, cmd)
            else:
                status = run_command(settings, cmd, log_file)
            cmd = ['make', '-C', cover_page_dir]
            if settings.get('log_level') == 'DEBUG':
                status = run_command(settings, cmd)
            else:
                status = run_command(settings, cmd, log_file)
            cover_pdf = ''.join(glob.glob(cover_page_dir + '/*.pdf'))
            if os.path.isfile(cover_pdf):
                _, filename = os.path.split(latex_pdf)
                pdfs = [cover_pdf, latex_pdf]
                merger = PdfFileMerger()
                for pdf in pdfs:
                    merger.append(pdf)
                merger.write(filename)
                logger.debug("Merged MeerKAT cover page with %s"%filename)
                os.rename(filename, '/'.join([latex_dir, filename]))
                logger.info("Moved %s back into build directory"%filename)


        now = time.localtime()
        build_dir = settings["build_dir"]
        katreport_dir = settings["katreport_dir"]
        # Make run_tests directory
        dirName = 'CBF_Tests_Reports'
        logger.debug("Creating directory ../%s" % dirName)
        cmd = ['mkdir', '-p', '../%s' % dirName]
        status = run_command(settings, cmd, log_file)
        if status:
            logger.error("there was an error n mkdir -p ../%s" % dirName)
        logger.debug("copy build to ../%s/%s" % (dirName, build_dir))
        cmd = ['cp', '-r', 'build', '../%s/%s' % (dirName, build_dir)]
        status = run_command(settings, cmd, log_file)
        if status:
            logger.error("there was an error on copying build to ../%s/%s" % (dirName, build_dir))
        # Copy katreport_dir directory
        logger.debug("copy ./%s to ../%s/%s" % (katreport_dir, dirName, build_dir))
        cmd = ['cp', '-r', katreport_dir, "../%s/%s" % (dirName, build_dir)]
        status = run_command(settings, cmd, log_file)
        if status:
            logger.error("there was an error on copying ./%s to %s" % (katreport_dir, build_dir))
        else:
            logger.debug('Cleaning up %s directory' % docs_dir)
            run_command(settings, ['git', 'checkout', '--', docs_dir], log_file)
            logger.info('******Done generating the document!!!******')


def run_nose_test(settings):
    """
    Run the nose test:
    output is captured in <katreport_dir>/output.log
    result is captured in <katreport_dir>/katreport.json
    """
    os.chdir(settings['base_dir'])
    # try:
    #     cmd = [sh.which('nosetests')]
    #     assert cmd is not None
    # except Exception as e:
    #     cmd = ['/usr/local/bin/nosetests']
    cmd = ['/usr/local/bin/nosetests']
    katreport_dir = settings.get('katreport_dir')

    if settings.get('log_level'):
        cmd.append('-v')
        cmd.append('-s')
        cmd.append("--with-xunit")
        cmd.append("--xunit-file=%s/nosetests.xml" % katreport_dir)
    cmd.append("--with-katreport")
    if settings.get('use_core_json') and settings.get('json_file'):
        cmd.append("--katreport-requirements=%s" % settings['json_file'])

    # Build the nosetests filter.
    condition = {'OR': ['aqf_system_all'], 'AND': []}
    # if settings.get('system_type'):
    #     # Include tests for this system type.
    #     condition['OR'].append("aqf_system_%s" % settings['system_type'])

    _site_location = 'Karoo'
    if settings.get('system_location', '').lower().startswith(_site_location.lower()):
        # Disable intrusive if in the Karoo.
        condition['AND'].append("(not aqf_intrusive)")
    else:
        # Disable site_only if not in karoo.
        condition['AND'].append("(not aqf_site_only)")

    if settings.get('site_acceptance'):
       if settings.get('system_location', '').lower().startswith(_site_location.lower()):
           # Include site_only tests if in the Karoo
           condition['AND'].append("(aqf_site_acceptance or aqf_site_only)")
       else:
           condition['AND'].append("aqf_site_acceptance")

    if settings.get('mode'):
        _instrument = settings.get('mode')
        _decorated_instrument = 'aqf_instrument_%s'%_instrument
        if settings.get('site_acceptance'):
            # For Acceptance
            # run tests decorated with aqf_instrument_MODE and site_acceptance
            # and aqf_site_tests
            _conditions = '(aqf_site_test or (aqf_site_acceptance and %s))' % _decorated_instrument
            condition['AND'].append(_conditions)
        elif ((_instrument.startswith('bc16') or _instrument.startswith('bc32')) and \
            settings['system_location'].lower() == 'lab'):
            logger.error("Test can ONLY be ran on SITE!!!!!!!!")
            sys.exit(1)
        else:
            # run only tests decorated with aqf_instrument_MODE
            condition['AND'].append(_decorated_instrument)
    else:
        # Not demo mode - thus this is AUTO testing
        if settings.get('site_acceptance'):
            # For Acceptance Testing:
            # run aqf_auto_test decorated with site_acceptance
            # and aqf_demo_tests decorated with site_acceptance
            # and aqf_site_tests
            condition['AND'].append('(aqf_site_test or aqf_site_acceptance)')
        else:
            # For Qualification Testing:
            # run all tests except site_tests
            condition['AND'].append('(not aqf_site_test)')
            pass

    if not settings.get('slow_test'):
        condition['AND'].append("not aqf_slow")
    # , 'aqf_generic_test'
    # Mix OR with AND
    condition['or_str'] = ' or '.join(condition['OR'])
    if condition['AND'] and condition['or_str']:
        condition['AND'].append("(%s)" % condition['or_str'])
        if condition['AND'][-1].startswith('('):
            try:
                assert settings['mode'] is None
                condition['AND'][-1] = condition['AND'][-1].replace('(','').replace(')','')
            except AssertionError:
                condition['AND'][-1] = condition['AND'][-1].replace('(','')
        if not condition['AND'][-2].startswith('('):
            condition['AND'][-2] = '(' + condition['AND'][-2]
        cmd.append('-A(%s)' % ' and '.join(condition['AND']))
    elif condition['AND']:
        cmd.append('-A(%s)' % ' and '.join(condition['AND']))
    elif condition['or_str']:
        cmd.append('-A(%s)' % condition['or_str'])
    cmd.append('-A(%s)' % 'aqf_generic_test')

    katreport_control = []
    if settings.get('jenkins'):
        katreport_control.append('jenkins')
    # if settings.get('demo'):
    #     katreport_control.append('demo')
    if settings.get('katreport_quick'):
        katreport_control.append('quick')
    if katreport_control:
        cmd.append("--katreport-control=%s" % ','.join(katreport_control))

    if settings.get('site_acceptance') or settings.get('site'):
        # Use different directory for acceptance results, so as not to overwrite qualification results
        cmd.append("--katreport-name=katreport_acceptance") # Using default "katreport" for qualification

    # if settings.get('jenkins'):
    #     cmd.append("--with-xunit")
    #     cmd.append("--xunit-file=build/nosetests.xml")

    if settings.get('tests'):
        cmd.append(settings.get("tests"))
    else:
        logger.error("File containing tests not found!")

    nose_args = settings.get('nose_args', '')
    if nose_args:
        for arg in nose_args.split():
            cmd.append(arg)
        # Run with --logging-level WARN if logging-level not passed in with nose_args
        cmd.append("--logging-level=WARN")
    else:
        cmd.append("--logging-level=INFO")

    # Let the output log be written into the katreport_dir
    cmd.append(" 2>&1 | tee %s/output.log" % (katreport_dir))
    logger.debug("Running nosetests with following command: %s" % cmd)
    return run_command(settings, cmd, shell=True)

def _downloadfile(url, filename):
    from urllib2 import urlopen, HTTPError
    logger.info("Download {} from {}".format(filename, url))
    try:
        stream = urlopen(url)
        with open(filename, 'w') as fh:
            fh.write(stream.read())
    except HTTPError as e:
        logger.error('%s %s' % (e, url))
        filename = None
    return filename

def get_filename(what, settings):
    katreport_dir = settings["katreport_dir"]
    files = {
                'test': os.path.join(settings['me_dir'],
                                  '{}/katreport.json'.format(katreport_dir)),
                'system': os.path.join(settings['me_dir'],
                                    '{}/katreport_system.json'.format(katreport_dir)),
                'core': settings.get('json_file', '')
             }
    return files.get(what, None)

def generate_report(settings):
    report_type = settings['report'].lower()
    katreport_dir = settings["katreport_dir"]
    files = {
                'test': get_filename('test', settings),
                'system': get_filename('system', settings),
                'core': get_filename('core', settings)
            }
    urls_for_jenkins = {
                            'test': "katreport.json",
                            'system': "katreport_system.json"
                        }

    # TODO(MS) Get this from the command line so that it can be passed in for
    # each different document generation.
    if report_type == 'jenkins':
        url_server = "http://dbelab04:8080"
        url_path1 = "view/CBF/view/CBF%20Devel"
        url_path2 = "job/mkat_fpga_tests/lastSuccessfulBuild"
        url_path3 = "artifact/katreport"
        print "Fetch from jenkins"
        logger.info("Fetch files from jenkins")
        for item in urls_for_jenkins:
            url = '/'.join([url_server, url_path1, url_path2, url_path3,
                            urls_for_jenkins[item]])
            tmp_filename = os.path.join('/tmp', 'aqf_' + item + '.json')
            files[item] = _downloadfile(url, tmp_filename)

    for f in files:
        if not files[f] or not os.path.isfile(files.get(f, '')):
            logger.error("The {0} data file {1} could not be found.".
                     format(f, files[f]))

    report = Report(system_data=files['system'],
                    acceptance_report=settings.get('site_acceptance'))
    report.load_core_requirements(files['core'])
    report.load_test_results(filename=files['test'])

    report_name = "katreport"
    reports = {}
    reports['core'] = 'katreport_core.rst'
    reports['system'] = 'katreport_system.rst'
    report.base_dir = os.path.join(settings['me_dir'], katreport_dir)
    for report_type in reports:
        report.clear()
        filename = os.path.join(settings['me_dir'], katreport_dir,
                                reports[report_type])
        report.write_to_file(filename, report_type)

    report.clear()
    report.write_rst_cbf_files(os.path.join(settings['me_dir'], katreport_dir),
                               settings['build_dir'], settings["katreport_dir"], 'cbf')

def show_test_results(settings):
    """Helper function to print test results

    test_data['Meta'] typically
        {u'end': u'2015-08-18 05:30:17.308133',
         u'sitename': u'devl_mkat',
         u'start': u'2015-08-18 04:33:30.186272',
         u'sys_args': [u'/usr/local/bin/nosetests',
         u'--with-katreport',
         u'--katreport-requirements=/tmp/CORE_EXPORT/M.json',
         u'-A((not aqf_site_only) and (not aqf_site_test) and (aqf_system_all or aqf_system_mkat))',
         u'./tests',
         u'--logging-level=WARN']}

    For each test_data[test] there is a dictionary with typically the following entries:
         u'status', u'group', u'description', u'success', u'label', u'steps', u'systems',
         u'requirements', u'error_msg'
         Optional:
            u'demo' (to be deprecated soon)
            u'aqf_demo_test', u'aqf_auto_test', u'aqf_site_test'
            u'aqf_site_only', u'aqf_site_acceptance',
            u'aqf_slow'
            u'aqf_intrusive' (to be deprecated soon)

    """

    #from report_generator.report import Report
    #report = Report(system_data=files['system'],
    #                acceptance_report=settings.get('site_acceptance'))
    #report.load_core_requirements(files['core'])
    #report.load_test_results(filename=files['test'])
    #report.show_test_results()
    #or report.show_core_requirements()

    filename = get_filename('test', settings)

    with open(filename, 'r') as fh:
        test_data = json.loads(fh.read())

    print "===Meta==="
    for item in sorted(test_data["Meta"].keys()):
        print "    ",item.ljust(20),": ",test_data["Meta"][item]

    tests = [item for item in test_data.keys() if item != 'Meta']
    for test in tests:
        print "\n-------",test,"------"
        for item in sorted(test_data[test].keys()):
            if item not in ['steps','description']:
                print "    ",item.ljust(20),": ",test_data[test][item]
        # IF verbose, print description and steps at the end
        if settings["verbose"]:
            for item in ['description', 'steps']:
                print "    ",item.ljust(20),": ",test_data[test][item]

def do_cleanup(settings):
    """Run make clean and remove previous run."""
    katreport_dir = settings["katreport_dir"]
    logger.info('Remove HTML files and results from previous tests from %s' % katreport_dir)
    os.chdir(settings.get('me_dir'))
    run_command(settings, ['make', 'clean'])
    run_command(settings, ['rm', '-rf', katreport_dir])
    try:
        os.makedirs(katreport_dir)
    except os.error:
        pass

def gather_system_settings(settings):
    """Get information of the system we are running on."""
    try:
        from getpass import getuser
        get_username = getuser()
    except OSError:
        import pwd
        get_username = pwd.getpwuid(os.getuid()).pw_name

    filename = os.path.join(settings['me_dir'], settings["katreport_dir"], 'katreport_system.json')
    data = {
                # 'pip_version':  {
                #                 },
                # 'dpkg_version': {
                #                 },
                # # 'vcs_version':  {
                #                 }
            }
    _items = ['site', 'system_type', 'system_location', 'system_config']
    for item in _items:
        data[item] = settings[item]

    # cmd = ['pip', 'freeze']
    # for line in str.splitlines(run_command_output(settings, cmd)):
    #     package = {}
    #     if not line.startswith("#"):
    #         segments = line.split("==")
    #         package['name'] = segments[0]
    #         package['ver'] = " ".join(segments[1:2])
    #         #data['pip_version'][package['name']] = package['ver']

    # cmd = ['dpkg', '-l']
    # for line in str.splitlines(run_command_output(settings, cmd)):
    #     package = {}
    #     if line.startswith("i"):
    #         line_segments = line.split()
    #         #data['dpkg_version'][line_segments[1]] = line_segments[2]

    data['dist'] = ' '.join(platform.dist())
    data['uname'] = ' '.join(os.uname())
    data['username'] = get_username.upper()
    # data['environment'] = os.environ
    # data['environment'] = dict([(i, str(data['environment'].get(i, '')))
    #                             for i in data['environment']])
    cmd = ['ip', '-f', 'inet', 'addr']
    data['ip_addresses'] = str.splitlines(run_command_output(settings, cmd))
    # Here we define the labels that katreport will use. Dont want katreport
    # to have to much knowledge of this file.
    data['Labels'] = {
                        # 'pip_version': {
                        #                     'label': 'Python Modules',
                        #                     'description': 'Versions of python '
                        #                     'modules managed by pip. This data was '
                        #                     'obtained by running "pip freeze."'
                        #                 },
                        # 'vcs_version': {
                        #                     'label': 'Versions from VCS',
                        #                     'description': 'Output from '
                        #                     'kat-versioncontrol.py'
                        #                 },
                        # 'dpkg_version': {
                        #                     'label': 'Installed Software',
                        #                     'description': 'Versions of software '
                        #                     'installed on this system.'
                        #                 },
                        # 'username':     {
                        #                     'label': 'Unix User',
                        #                     'description': 'The name of the user who executed the '
                        #                     'tests:'
                        #                 },
                        'system_location': {
                                            'label': 'Test Configuration - Hardware',
                                            'description': 'Hardware information: '
                                        },
                        'system_config': {
                                            'label': 'Test Configuration - Software',
                                            'description': 'Software Information: ',
                                        },
                        'system_type': {
                                            'label': 'CBF Instrument Under Test',
                                            'description': 'The name of the instrument that was ran:'
                                        },

                        # 'dist':         {
                        #                     'label': 'OS Distribution',
                        #                     'description': 'Distribution of the Operating System:'
                        #                 },
                        # 'uname':        {
                        #                     'label': 'Linux Kernel',
                        #                     'description': 'Current Linux Kernel:'},
                        #                     'ip_addresses': {
                        #                                         'label': 'IP Adresses',
                        #                                         'description': 'IP Adresses of the '
                        #                                         'system test was ran on:'
                        #                                     }
                     }
    with open(filename, 'w') as fh:
        fh.write(json.dumps(data, indent=4))

def verify_dependecies(module_name):
    """
    Check if all module dependencies are satisfied.
    """
    try:
        _None, _module_loc, _None = __import__('imp').find_module(module_name)
        logger.debug("%s has been installed, and can be located in %s" % (module_name, _module_loc))
    except ImportError:
        logger.error("Test dependency module missing, please reinstall %s." % module_name)
        sys.exit(1)

def kill_pid(proc_name):
    """
    Retrieve process pid and send kill signal
    """
    try:
        os.kill(int(subprocess.check_output(["pgrep", proc_name])), SIGKILL)
    except Exception:
        pass

def plot_backend(_filename, oldcontent, newcontent):
    """
    matplotlib backend support
    """
    current_user = pwd.getpwuid(os.getuid())[0]
    if current_user == 'cbf-test':
        try:
            with open(_filename,'r') as f:
                newlines = []
                for line in f.readlines():
                    newlines.append(line.replace(oldcontent, newcontent))
            with open(_filename, 'w') as f:
                for line in newlines:
                    f.write(line)
        except Exception:
            pass

def PyCompile(settings):
    """
    Compile python file, if encounter errors exit.
    """
    try:
        py_compile.compile(settings.get('tests', False), doraise=True)
    except Exception, e:
        logger.exception('Failed to run the python compiler')
        sys.exit(1)

def katcp_request(port, katcprequest='help', timeout=10):
    """
    Katcp requests on certain port
    port: int
    katcprequest: str
    timeout: int
    """
    client = katcp.BlockingClient('127.0.0.1', port)
    client.setDaemon(True)
    client.start()
    is_connected = client.wait_connected(timeout)
    if not is_connected:
      client.stop()
      #raise RuntimeError('Could not connect to corr2_servlet, timed out.')
      return
    try:
        reply, informs = client.blocking_request(katcp.Message.request(katcprequest),
            timeout=timeout)
    except Exception:
        return None
    client.stop()
    client = None
    if reply.reply_ok():
        return informs

def get_running_instrument():
    """
    Using katcp: Retrieve running instrument if available
    """
    try:
        _katcp_req = katcp_request(7147, 'array-list')
        assert isinstance(_katcp_req, list)
        katcp_client_port = [int(i.arguments[1]) for i in _katcp_req
                             if i.arguments[0].startswith('arr')][0]
        assert isinstance(katcp_client_port, int)
    except Exception:
        return False
    else:
        _katcp_req = katcp_request(katcp_client_port, 'sensor-value')
        sensors_required = ['instrument-state']
        sensors = {}
        srch = re.compile('|'.join(sensors_required))
        for inf in _katcp_req:
            if srch.match(inf.arguments[2]):
                sensors[inf.arguments[2]] = inf.arguments[4]
        if sensors:
            return sensors.values()

def get_version_list():
    """Using katcp: Retrieve CBF information"""
    try:
        katcp_array_list = katcp_request(7147, 'array-list')
        assert isinstance(katcp_array_list, list)
        katcp_port = [int(i.arguments[1]) for i in katcp_array_list
                 if i.arguments[0].startswith('arr')][0]
        assert isinstance(katcp_port, int)
    except Exception:
        return False
    else:
        return katcp_request(katcp_port, 'version-list')

def generate_index(document):
    """Sphinx-build index.rst generator"""
    def _writer(document):
        file = 'index.rst'
        with open(file, 'w') as f:
            f.write(document)
    if document:
        logger.debug('Writing data to index.rst file:\n%s' % document)
        _writer(document)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    options, args = option_parser()
    log_level = None
    if options.log_level:
        log_level = options.log_level.strip()
        try:
            import logging
            logging.getLogger('katcp').setLevel(logging.ERROR)
            logging.basicConfig(level=getattr(logging, log_level))
            logger = logging.getLogger(__file__)
            coloredlogs.install(level=log_level, logger=logger,
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(pathname)s : '
                   '%(lineno)d - %(message)s')
        except AttributeError:
            raise RuntimeError('No such log level: %s' % log_level)

    plot_backend('matplotlibrc', 'TKagg', 'agg')
    kill_pid('nosetests')
    settings = dict((k, getattr(options, k)) for k in dir(options)
                    if not callable(getattr(options, k))
                    and not k.startswith('_'))
    settings.update(get_system_info())
    try:
        settings['system_type'] = ''.join(get_running_instrument())
        settings['system_config'] = [': '.join(i.arguments) for i in get_version_list()]
        assert settings['system_config']
    except TypeError:
        settings['system_type'] = None
    except AssertionError:
        settings['system_config'] = 'Unknown'

    settings['process_core'] = True
    settings['gather_system_settings'] = True
    settings['tests'] = args
    settings['me'] = os.path.abspath(__file__)
    settings['me_dir'] = os.path.dirname(settings['me'])
    settings['src_dir'], settings['test_dir'] = os.path.split(settings['me_dir'])
    if not (settings['test_dir'] == 'mkat_fpga_tests'):
        settings['test_dir'] = 'mkat_fpga_tests'
    settings['core_backup'] = '/usr/local/src/core_export/'
    settings['scripts'] = '/'.join([settings['me_dir'], 'scripts'])
    test_class = 'test_CBF'
    settings['tests_class'] = test_class
    try:
        test_file = max(glob.iglob(
            os.path.join(settings['me_dir'], settings['test_dir'], '[Tt][est_cbf.py]')),
            key=os.path.getctime)
        settings['tests'] = test_file
    except Exception:
        settings['tests'] = os.path.join(settings['me_dir'], settings['test_dir'], 'test_cbf.py')

    if settings["site_acceptance"]:
        settings['katreport_dir'] = "katreport_acceptance"
    else:
        settings['katreport_dir'] = "katreport"
    if 'base_dir' not in settings:
        settings['base_dir'] = os.getcwdu()

    os.chdir(settings['me_dir'])
    if not os.path.exists(settings['katreport_dir']):
        os.mkdir(settings['katreport_dir'])

    if settings.get('katreport_quick'):
        settings['gather_system_settings'] = False
        settings['process_core'] = False
        settings['slow_test'] = False
        settings['report'] = 'skip'

    # if settings.get('demo'):
    #     settings['report'] = 'skip'
    #     settings['gather_system_settings'] = False
    # if settings.get('dry_run'):
    #     settings['report'] = 'skip'

    # if settings.get('jenkins'):
    #     settings['dev_update'] = True
    #     settings['cleanup'] = True
    #     settings['slow_test'] = True
    #     settings['report'] = 'skip'
    if settings['report'] in ['jenkins']:
        settings['gather_system_settings'] = False

    if settings['report'] in ['skip']:
        settings['process_core'] = False
        settings['gen_html'] = False

    # Do the different steps.
    log_func = create_log_func(settings)

    if os.path.exists(settings.get('scripts')):
        cmdPath = ''.join([i for i in glob.glob('scripts/*') if 'instrument_activate' in i])
        if settings['mode'] and settings['instrument_activate'] and os.path.isfile(cmdPath):
            logger.info('Starting and instrument: %s' %settings['mode'])
            cmd = ['bash', 'scripts/instrument_activate', settings.get('mode')]
            # Allow instrument to be activated in 200 seconds
            timeout = 200
            initInstrument = RunCmdTimeout(settings, cmd, timeout)
            initInstrument.daemon = True
            status = initInstrument.run_the_process()
            if status is 0:
                logger.info('Instrument %s activated ok!'%settings.get('mode'))
            else:
                msg = 'Failed to initialise %s instrument!'%settings.get('mode')
                logger.error(msg)
                raise RuntimeError(msg)

    PyCompile(settings)

    if settings.get('dev_update'):
        do_dev_update(settings)

    if settings.get('cleanup'):
        do_cleanup(settings)

    if settings['gather_system_settings']:
        for module_name in _core_dependencies:
            verify_dependecies(module_name)
        gather_system_settings(settings)

    if settings['process_core']:
        process_core_data(settings)
    now = time.localtime()
    start_time = ("%02d%02d%02d-%02dh%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour,
                  now.tm_min))
    if settings.get('mode'):
        settings['build_dir'] = "%s-" % settings.get('mode') + start_time
    else:
        settings['build_dir'] = "%s-%s" % (settings.get('system_type', 'Unknown'), start_time)
        # settings['mode'] = settings.get('system_type', 'Unknown')
        # settings['build_dir'] = "%s-" % settings.get('mode') + start_time

    if settings.get('log_level') == 'DEBUG':
        print "=========settings========="
        for key in settings:
            print key,":",settings[key]
        print "=========================="

    condition = ((settings['report'] in ['local_&_test', 'skip'] or settings.get(
                 'dry_run')) and not settings.get('cleanup'))
    if condition:
        run_nose_test(settings)
    if settings['report'] in ['results']:
        show_test_results(settings)
    elif settings['report'] not in ['skip']:
        try:
            if settings.get('gen_html', False) or settings.get('gen_qtp', False) or settings.get(
                'gen_qtr', False):
                generate_report(settings)
                generate_sphinx_docs(settings)
        except Exception as e:
            errmsg = "Experienced some issues: %s" % str(e)
            logger.error(errmsg)
            sys.exit(errmsg)