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

import glob
import json
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
from report_generator.report import Report
from shutil import copyfile
from signal import SIGKILL

# List all core test python module dependencies
_core_dependencies = ['corr2', 'casperfpga', 'spead2', 'katcp']

def option_parser():
    usage = """
        Usage: %prog [options]
        This script auto executes CBF Tests with selected arguments.
        See Help for more information.
        """
    parser = OptionParser(usage)
    parser.add_option("-v", "--verbose",
                      action="store_true",
                      dest="verbose",
                      help="Be more verbose")

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

    parser.add_option("--no_html",
                      dest="gen_html",
                      action="store_false",
                      default=True,
                      help="Do not generate the html output")

    parser.add_option("--with_pdf",
                      dest="gen_pdf",
                      action="store_true",
                      default=False,
                      help="Generate PDF report output")

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

def run_command(settings, log_func, cmd, log_filename=None, stdout=False, stderr=False, shell=False):
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
        else:
            raise RuntimeError('nose is not installed.')

    if settings.get('dry_run'):
        # Just print the command
        log_func('INFO', *cmd)
        os.environ['DRY_RUN'] = 'True'

    if settings.get('available-tests'):
        # nosetests -vv collect-only
        if cmd[0].endswith('nosetests'):
            cmd.insert(1, '--collect-only')
            log_func('INFO', *cmd)

    if log_filename and not stdout and not stderr:
        with open(log_filename, 'w') as fh:
            log_func('DEBUG', 'Writting %s to file: %s' %(' '.join(cmd), log_filename))
            return subprocess.call(cmd, stderr=fh, stdout=fh)
    else:
        log_func('DEBUG', 'Run command with stderr:', *cmd)
        kwargs_cmd = {'env': os.environ}
        try:
            if shell:
                kwargs_cmd['shell'] = True
                str_cmd = " ".join(cmd)
                log_func('DEBUG', 'Run command with shell', str_cmd)
                return subprocess.call(str_cmd, **kwargs_cmd)
            else:
                return subprocess.call(cmd, **kwargs_cmd)
        except KeyboardInterrupt as e:
            kill_pid('nosetests')
            msg = "Test closed prematurely, and process has since been killed"
            raise RuntimeError(msg)


def run_command_output(settings, log_func, cmd):
    """Run a command and return output."""
    if settings.get('dry_run'):
        log_func('INFO', *cmd)
        return ''
    else:
        try:
            return subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=open('/dev/null', 'a')
                                    ).communicate()[0]
        except OSError:
            return ''


class RunCmdTimeout(threading.Thread):
    def __init__(self, settings, log_func, cmd, timeout):
        """
        Run a command with timeout
        """
        threading.Thread.__init__(self)
        self.cmd = cmd
        self.timeout = timeout

    def run(self):
        if settings.get('dry_run'):
            log_func('INFO', *cmd)
            return ''
        else:
            try:
                if settings['verbose']:
                    self.p = subprocess.Popen(self.cmd)
                else:
                    self.p = subprocess.Popen(self.cmd, stdout=open('/dev/null', 'a'),
                        stderr=subprocess.PIPE)
                self.p.wait()
            except OSError:
                log_func('ERROR', 'OSError: Failed to execute command')
                return ''

    def run_the_process(self):
        try:
            self.start()
            self.join(self.timeout)
        except KeyboardInterrupt:
            log_func('ERROR', 'KeyboardInterrupt')
        if self.is_alive():
            self.p.terminate()
            self.join()
            return self.p.returncode
        else:
            try:
                return self.p.returncode
            except Exception:
                log_func('ERROR', 'Failed to execute command')
                return


def do_dev_update(settings, log_func):
    """Do a code update and install."""
    log_func("DEBUG", "Will perform a Python package update")
    os.chdir(settings.get('me_dir'))
    _pip = sh.which('pip')
    run_command(settings, log_func, [_pip, 'install', '-U', 'pip'])
    run_command(settings, log_func, [_pip, 'install', '-U', '-r', 'pip-requirements.txt'])

def get_system_info():
    _system = os.uname()
    system = {'systype': '', 'site': '', 'nodetype': 'jenkins_build_slave'}
    system['sysname'] = os.uname()[1]
    # Backwards compatability.
    if _system[1].startswith('cmc'):
        system['site'] = True
        system['system_location'] = 'Karoo'
    elif _system[1].startswith('dbe'):
        system['site'] = False
        system['system_location'] = 'Lab'
    else:
        # Unknown system location, and assuming it is not on site
        system['site'] = False
        system['system_location'] = None
    system['system_type'] = system.get('systype')
    return system

def process_core_data(settings, log_func):
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
    log_func('INFO', 'Processing CORE Data')
    # Update SVN.
    if not os.path.isdir(settings['tmp_core_dir']):
        os.mkdir(settings['tmp_core_dir'])
    os.chdir(settings['tmp_core_dir'])
    try:
        core_supplemental_dir = os.path.join(settings['me_dir'], 'supplemental', "MeerKAT.xml")
        settings['xml_file'] = core_supplemental_dir
        assert os.path.exists(core_supplemental_dir)
    except AssertionError as e:
        log_func('WARNING', 'CORE.xml does not exist in directory: %s' % core_supplemental_dir)
        core_backup = settings.get('core_backup')
        log_func('INFO', 'Retrieving CORE.xml from backup dir: %s' % core_backup)
        if os.path.exists(core_backup):
            latest_core_xml = max(glob.iglob(os.path.join(core_backup ,'*.[Xx][Mm][Ll]')),
                key=os.path.getctime)
            log_func('INFO', 'CORE.xml file size %.2f Mb'%(os.path.getsize(latest_core_xml) / 1e6))
            settings['xml_file'] = latest_core_xml
        else:
            errmsg = 'CORE.xml file does not exist in %s dir'%core_backup
            log_func('ERROR', errmsg)
            raise RuntimeError(errmsg)

    if os.path.isfile(settings['xml_file']):
        settings['xml_mtime'] = os.path.getmtime(settings['xml_file'])

    if os.path.isfile(settings['json_file']):
        settings['json_mtime'] = os.path.getmtime(settings['json_file'])

    # Update JSON
    if (settings['xml_mtime'] > 0 and
            settings['json_mtime'] < settings['xml_mtime']):
        log_func("Process: XML -> JSON")
        if not settings.get('dry_run'):
            process_xml_to_json(settings['xml_file'], settings['json_file'], verbose=True,
                log_func=log_func)
    else:
        log_func("JSON File is up-to-date")
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

def generate_html_sphinx_docs(settings, log_func):
    os.chdir(settings['base_dir'])
    log_file = '/dev/null'
    cmd = ['make', 'html']
    try:
        assert settings['gen_html']
    except AssertionError:
        pass
    else:
        log_func("INFO", "Generating HTML document from rst files")
        if settings['verbose']:
            status = run_command(settings, log_func, cmd)
        else:
            status = run_command(settings, log_func, cmd, log_file)

    cmd = ['make', 'latexpdf']
    try:
        assert settings['gen_pdf']
    except AssertionError:
        pass
    else:
        log_func("INFO", "Generating PDF document from rst files")
        copyfile('report_generator/index.rst', 'index.rst')
        if settings['verbose']:
            status = run_command(settings, log_func, cmd)
        else:
            status = run_command(settings, log_func, cmd, log_file)
        log_func("DEBUG", "Undo changes made to index.rst to accomodate PDF generation")
        cmd = ['git', 'checkout', '--', 'index.rst']
        status = run_command(settings, log_func, cmd, log_file)

    if status:
        log_func("ERROR", "there was an error on 'make html' - not copying build results")
    else:
        now = time.localtime()
        build_dir = settings["build_dir"]
        katreport_dir = settings["katreport_dir"]
        # Make run_tests directory
        dirName = 'CBF_Tests_Reports'
        log_func("INFO", "Creating directory ../%s"%dirName)
        cmd = ['mkdir', '-p', '../%s'%dirName]
        status = run_command(settings, log_func, cmd, log_file)
        if status:
            log_func("ERROR", "there was an error n mkdir -p ../%s"%dirName)
        # Copy build directory
        log_func("DEBUG", "Text color mappings")
        cmd = ['bash', 'scripts/generate_color.sh']
        run_command(settings, log_func, cmd, log_file)
        log_func("DEBUG", "copy build to ../%s/%s" % (dirName, build_dir))
        cmd = ['cp', '-r', 'build', '../%s/%s' % (dirName, build_dir)]
        status = run_command(settings, log_func, cmd, log_file)
        if status:
            log_func("ERROR", "there was an error on copying build to ../%s/%s" %(dirName, build_dir))
        # Copy katreport_dir directory
        log_func("DEBUG", "copy ./%s to ../%s/%s"%(katreport_dir, dirName, build_dir))
        cmd = ['cp', '-r', katreport_dir, "../%s/%s"%(dirName, build_dir)]
        status = run_command(settings, log_func, cmd, log_file)
        if status:
            log_func("ERROR", "there was an error on copying ./%s to %s"%(katreport_dir, build_dir))

def run_nose_test(settings, log_func):
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
    # Note settings['verbose'] is a tri-state, where none is normal
    # verbosity level. True is more verbose and False is less.
    if settings['verbose'] is True:
        cmd.append('-v')
        cmd.append('-s')
        cmd.append("--with-xunit")
        cmd.append("--xunit-file=%s/nosetests.xml"%katreport_dir)
    elif settings['verbose'] is False:
        cmd.append('-q')
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
            condition['AND'].append('(aqf_site_test or (aqf_site_acceptance and %s))'%_decorated_instrument)
        elif ((_instrument.startswith('bc16') or _instrument.startswith('bc32')) and \
            settings['system_location'].lower() == 'lab'):
            log_func("ERROR", "Test can ONLY be ran on SITE!!!!!!!!")
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
        log_func("ERROR", "File containing tests not found!")

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
    return run_command(settings, log_func, cmd, shell=True)

def _downloadfile(url, filename, log_func):
    from urllib2 import urlopen, HTTPError
    log_func("INFO", "Download {} from {}".format(filename, url))
    try:
        stream = urlopen(url)
        with open(filename, 'w') as fh:
            fh.write(stream.read())
    except HTTPError as e:
        log_func("ERROR", e, url)
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

def generate_report(settings, log_func):
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
        log_func('INFO', "Fetch files from jenkins")
        for item in urls_for_jenkins:
            url = '/'.join([url_server, url_path1, url_path2, url_path3,
                            urls_for_jenkins[item]])
            tmp_filename = os.path.join('/tmp', 'aqf_' + item + '.json')
            files[item] = _downloadfile(url, tmp_filename, log_func)

    for f in files:
        if not files[f] or not os.path.isfile(files.get(f, '')):
            log_func('ERROR', "The {0} data file {1} could not be found.".
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
                               settings['build_dir'], settings["katreport_dir"],
                               'cbf')

def show_test_results(settings, log_func):
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

def do_cleanup(settings, log_func):
    """Run make clean and remove previous run."""
    katreport_dir = settings["katreport_dir"]
    log_func('INFO', 'Remove HTML files and results from previous tests from %s' % katreport_dir)
    os.chdir(settings.get('me_dir'))
    var = raw_input("Are you sure you want to delete contents in %s:" % katreport_dir)
    if var == ('y' or 'Y'):
        run_command(settings, log_func, ['make', 'clean'])
        run_command(settings, log_func, ['rm', '-rf', katreport_dir])
        try:
            os.makedirs(katreport_dir)
        except os.error:
            pass
    else:
        sys.exit(1)

def gather_system_settings(settings, log_func):
    """Get information of the system we are running on."""
    try:
        from getpass import getuser
        get_username = getuser()
    except OSError:
        import pwd
        get_username = pwd.getpwuid(os.getuid()).pw_name

    katreport_dir = settings["katreport_dir"]
    filename = os.path.join(settings['me_dir'], katreport_dir,
                            'katreport_system.json')
    data = {
                'pip_version':  {
                                },
                'dpkg_version': {
                                },
                'vcs_version':  {
                                }
            }

    for item in ['nodetype', 'site', 'systype',
                 'system_type', 'system_location']:
        data[item] = settings[item]

    cmd = ['pip', 'freeze']
    for line in str.splitlines(run_command_output(settings, log_func, cmd)):
        package = {}
        if not line.startswith("#"):
            segments = line.split("==")
            package['name'] = segments[0]
            package['ver'] = " ".join(segments[1:2])
            data['pip_version'][package['name']] = package['ver']

    cmd = ['dpkg', '-l']
    for line in str.splitlines(run_command_output(settings, log_func, cmd)):
        package = {}
        if line.startswith("i"):
            line_segments = line.split()
            data['dpkg_version'][line_segments[1]] = line_segments[2]

    data['dist'] = platform.dist()
    data['uname'] = os.uname()
    data['username'] = get_username
    data['environment'] = os.environ
    data['environment'] = dict([(i, str(data['environment'].get(i, '')))
                                for i in data['environment']])
    data['ip_addresses'] = str.splitlines(
        run_command_output(settings, log_func, ['ip', '-f', 'inet', 'addr']))
    # Here we define the labels that katreport will use. Dont want katreport
    # to have to much knowledge of this file.
    data['Labels'] = {
                        'pip_version': {
                                            'label': 'Python Modules',
                                            'description': 'Versions of python '
                                            'modules managed by pip. This data was '
                                            'obtained by running "pip freeze."'
                                        },
                        'vcs_version': {
                                            'label': 'Versions from VCS',
                                            'description': 'Output from '
                                            'kat-versioncontrol.py'
                                        },
                        'dpkg_version': {
                                            'label': 'Installed Software',
                                            'description': 'Versions of software '
                                            'installed on this system.'
                                        },
                        'dist':         {
                                            'label': 'OS Distribution',
                                            'description': 'Distribution of the '
                                            'Operating System'
                                        },
                        'uname':        {
                                            'label': 'Uname',
                                            'description': 'Output of the uname command'},
                                            'ip_addresses': {
                                                                'label': 'IP Adresses',
                                                                'description': 'IP Adresses of the '
                                                                'system test was ran on'
                                                            }
                    }
    with open(filename, 'w') as fh:
        fh.write(json.dumps(data, indent=4))

def verify_dependecies(module_name, log_func):
    """
    Check if all module dependencies are satisfied.
    """
    try:
        _None, _module_loc, _None = __import__('imp').find_module(module_name)
        log_func("DEBUG", "%s has been installed, and can be located in %s"%(module_name, _module_loc))
    except ImportError:
        log_func("ERROR", "Test dependency module missing, please reinstall %s."%module_name)
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

def PyCompile(settings, log_func):
    """
    Compile python file, if encounter errors exit.
    """
    try:
        log_func('DEBUG', 'Compiling a source file (%s) to byte-code' % settings['tests'])
        py_compile.compile(settings['tests'], doraise=True)
    except Exception, e:
        log_func('ERROR', str(e))
        sys.exit(1)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    options, args = option_parser()
    plot_backend('matplotlibrc', 'TKagg', 'agg')
    kill_pid('nosetests')
    settings = dict((k, getattr(options, k)) for k in dir(options)
                    if not callable(getattr(options, k))
                    and not k.startswith('_'))
    settings.update(get_system_info())
    # if settings['manual_systype']:
    #     settings['systype'] = settings['manual_systype']
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
            cmd = ['bash', 'scripts/instrument_activate', settings.get('mode')]
            # Allow instrument to be activated in 200 seconds
            timeout = 200
            initInstrument = RunCmdTimeout(settings, log_func, cmd, timeout)
            initInstrument.daemon = True
            status = initInstrument.run_the_process()
            if status is 0:
                log_func('INFO', 'Instrument %s activated ok!'%settings.get('mode'))
            else:
                msg = 'Failed to initialise %s instrument!'%settings.get('mode')
                log_func('ERROR', msg)
                raise RuntimeError(msg)

    PyCompile(settings, log_func)

    if settings.get('dev_update'):
        do_dev_update(settings, log_func)

    if settings.get('cleanup'):
        do_cleanup(settings, log_func)

    if settings['gather_system_settings']:
        for module_name in _core_dependencies:
            verify_dependecies(module_name, log_func)
        gather_system_settings(settings, log_func)

    if settings['process_core']:
        process_core_data(settings, log_func)
    now = time.localtime()
    start_time = ("%02d%02d%02d-%02dh%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour,
                  now.tm_min))
    if settings.get('mode'):
        settings['build_dir'] = "%s-"%settings.get('mode') + start_time
    else:
        settings['build_dir'] = "build-" + start_time

    if settings['verbose']:
        print "=========settings========="
        for key in settings:
            print key,":",settings[key]
        print "=========================="

    condition = ((settings['report'] in ['local_&_test', 'skip'] or settings.get(
                 'dry_run')) and not settings.get('cleanup'))
    if condition:
        run_nose_test(settings, log_func)
    if settings['report'] in ['results']:
        show_test_results(settings, log_func)
    elif settings['report'] not in ['skip']:
        try:
            generate_report(settings, log_func)
            if settings['gen_html']:
                generate_html_sphinx_docs(settings, log_func)
        except Exception as e:
            errmsg = "Experienced some issues: %s" % sys.exc_info()[0]
            log_func("ERROR", errmsg)
            sys.exit(errmsg)

