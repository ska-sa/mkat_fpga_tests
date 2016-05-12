#!/usr/bin/env python
import os
import json
import subprocess
import platform
import time

from optparse import OptionParser
from process_core_xml import process_xml_to_json


def option_parser():
    usage = "usage: %prog [options] [tests]"
    parser = OptionParser(usage)
    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose",
                      help="Be more verbose")
    parser.add_option("-q", "--quiet",
                      action="store_false", dest="verbose",
                      help="Be more quiet")
    parser.add_option("--acceptance", dest="site_acceptance",
                      action="store_true", default=False,
                      help="Will only run test marked '@site_acceptance' or linked to .SITE. VRs,"
                           " if in the Karoo then also @site_only tests")
    parser.add_option("--dry_run", dest="dry_run",
                      action="store_true", default=False,
                      help="Do a dry run. Print commands that would be called")
    parser.add_option("--no_html", dest="gen_html",
                      action="store_false", default=True,
                      help="Do not generate the html output")
    parser.add_option("--demo", dest="demo",
                      action="store_true", default=False,
                      help="Run the tests linked to .DEMO. and .SITE. VRs in demo mode. Wait for user input "
                           "at each Aqf.checkbox instance")
    parser.add_option("--quick", dest="katreport_quick",
                      action="store_true", default=False,
                      help="Only generate a small subset of the reports")
    parser.add_option("--no_slow", dest="slow_test",
                      action="store_false", default=True,
                      help="Exclude tests marked as @slow in this test run")
    parser.add_option("--clean", dest="cleanup",
                      action="store_true", default=False,
                      help="Cleanup reports from previous test run. Reports "
                      "are replaced by default without --clean. Clean is "
                      "useful with --quick to only generate the html of the "
                      "test run report")
    parser.add_option("--dev_update", dest="dev_update",
                      action="store_true", default=False,
                      help="do an 'svn up' and install noseplugin katreport "
                      "before running the tests")
    parser.add_option("--report", dest="report",
                      action="store", type="string", default='local_&_test',
                      help="Only generate the reports. No tests will be run.\n"
                      "Valid options are: local, jenkins, skip and results. 'results' will print the katreport[_accept].json test results")
    parser.add_option("--nose", dest="nose_args",
                      action="store", type="string", default='',
                      help="Additional arguments to pass on to nosetests. "
                      "eg --nose \"-x -s -v\"")
    parser.add_option("--jenkins", dest="jenkins",
                      action="store_true", default=False,
                      help="Run this command with the correct flags for"
                      "jenkins.")
    parser.add_option("--manual_systype", dest="manual_systype",
                      action="store", default=None,
                      help="Overwrite the system systype used for report"
                      "generation on jenkins.")

    (cmd_options, cmd_args) = parser.parse_args()
    return cmd_options, cmd_args


def run_command(settings, log_func, cmd,
                log_filename=None, stdout=False, stderr=False, shell=False):
    if log_filename is False:
        log_filename = '/dev/null'
    if settings.get('dry_run') or shell:
        # Fundge the command to add " when the command is
        # printed or run as string in shell.
        if '/usr/local/bin/nosetests' in cmd:
            for item in range(len(cmd)):
                if cmd[item].startswith("-A"):
                    break
            cmd[item] = cmd[item].replace('-A(', '-A"(') + '"'

    if settings.get('dry_run'):
        # Just print the command
        log_func('INFO', *cmd)
    else:
        if log_filename and not stdout and not stderr:
            with open(log_filename, 'w') as fh:
                log_func('DEBUG', 'Run command:', *cmd)
                return subprocess.call(cmd, stderr=fh, stdout=fh)
        else:
            log_func('DEBUG', 'Run command with stderr:', *cmd)
            kwargs_cmd = {'env': os.environ}
            if shell:
                kwargs_cmd['shell'] = True
                str_cmd = " ".join(cmd)
                log_func('DEBUG', 'Run command with shell', str_cmd)
                return subprocess.call(str_cmd, **kwargs_cmd)
            else:
                return subprocess.call(cmd, **kwargs_cmd)


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


def do_dev_update(settings, log_func):
    """Do a code update and install."""
    log_func("INFO", "Will perform a SVN update and "
             "pip install of nose plugin katreport")

    os.chdir(settings.get('me_dir'))
    run_command(settings, log_func, ['svn', 'up'])
    run_command(settings, log_func, ['sudo', 'pip', 'install',
                                     '--upgrade', 'nosekatreport/.'])


def get_system_info():
    system = {'systype': '', 'site': '', 'nodetype': 'jenkins_build_slave'}
    if os.path.isfile('/var/kat/node.conf'):
        with open('/var/kat/node.conf', 'r') as fh:
            system.update(json.loads(fh.read()))

    if os.path.isfile('/var/kat/sitename'):
        with open('/var/kat/sitename', 'r') as fh:
            sys_name = fh.read().strip().split("_")
        system['site'] = sys_name.pop(0)
        system['systype'] = "_".join(sys_name)

    if os.path.isfile('/var/kat/nodetype'):
        with open('/var/kat/nodetype', 'r') as fh:
            system['nodetype'] = fh.read().strip()

    # Backwards compatability.
    system['system_location'] = system.get('site')
    system['system_type'] = system.get('systype')
    return system


def process_core_data(settings, log_func):
    """Process the CORE XML file if JSON file is not there or old."""

    # Setup.
    if 'tmp_core_dir' not in settings:
        settings['tmp_core_dir'] = "/tmp/CORE_EXPORT"
    if 'xml_file' not in settings:
        settings['xml_file'] = os.path.join(settings['tmp_core_dir'],
                                            "svn/MeerKAT.xml")
    if 'json_file' not in settings:
        settings['json_file'] = os.path.join(settings['tmp_core_dir'],
                                             "M.json")
    settings['xml_mtime'] = 0
    settings['json_mtime'] = 0
    settings['use_core_json'] = False
    log_func('INFO', 'Process CORE Data')
    # Update SVN.
    if not os.path.isdir(settings['tmp_core_dir']):
        os.mkdir(settings['tmp_core_dir'])
    os.chdir(settings['tmp_core_dir'])
    settings['xml_file'] = os.path.join(settings['me_dir'],
                                        'supplemental',
                                        "MeerKAT.xml")
    if os.path.isfile(settings['xml_file']):
        settings['xml_mtime'] = os.path.getmtime(settings['xml_file'])

    if os.path.isfile(settings['json_file']):
        settings['json_mtime'] = os.path.getmtime(settings['json_file'])

    # Update JSON
    if (settings['xml_mtime'] > 0 and
            settings['json_mtime'] < settings['xml_mtime']):
        log_func("Process: XML -> JSON")
        if not settings.get('dry_run'):
            process_xml_to_json(settings['xml_file'], settings['json_file'],
                                verbose=True, log_func=log_func)
    else:
        log_func("JSON File is uptodate")
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
            print "++", level, "++", " ".join([str(s) for s in args])

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
    log_func("INFO", "make html from rst documents")
    log_file = '/dev/null'
    cmd = ['make', 'html']
    status = run_command(settings, log_func, cmd, log_file)
    if status:
        log_func("ERROR", "there was an error on 'make html' - not copying"
                          " build results")
    else:
        now = time.localtime()
        build_dir = settings["build_dir"]
        katreport_dir = settings["katreport_dir"]
        # Make run_tests directory
        log_func("INFO", "mkdir ../run_tests")
        cmd = ['mkdir', '-p', '../run_tests']
        status = run_command(settings, log_func, cmd, log_file)
        if status:
            log_func("ERROR", "there was an error n mkdir -p ../run_tests")
        # Copy build directory
        log_func("INFO", "copy build to ../run_tests/%s" % build_dir)
        cmd = ['cp', '-r', 'build', '../run_tests/%s' % build_dir]
        status = run_command(settings, log_func, cmd, log_file)
        if status:
            log_func("ERROR", "there was an error on copying build to ../run_tests/%s" %
                              build_dir)
        # Copy katreport_dir directory
        log_func("INFO", "copy ./{} to ../run_tests/{}".format(katreport_dir, build_dir))
        cmd = ['cp', '-r', katreport_dir, "../run_tests/{}".format(build_dir)]
        status = run_command(settings, log_func, cmd, log_file)
        if status:
            log_func("ERROR", "there was an error on copying ./{} to {}"
                              .format(katreport_dir, build_dir))

def run_nose_test(settings, log_func):
    """
    Run the nose test:
    output is captured in <katreport_dir>/output.log
    result is captured in <katreport_dir>/katreport.json
    """
    os.chdir(settings['base_dir'])
    cmd = ['/usr/local/bin/nosetests']
    katreport_dir = settings.get('katreport_dir')
    # Note settings['verbose'] is a tri-state, where none is normal
    # verbosity level. True is more verbose and False is less.
    if settings['verbose'] is True:
        cmd.append('-v')
        cmd.append('-s')
    elif settings['verbose'] is False:
        cmd.append('-q')
    cmd.append("--with-katreport")
    if settings.get('use_core_json') and settings.get('json_file'):
        cmd.append("--katreport-requirements=%s" % settings['json_file'])

    # Build the nosetests filter.
    condition = {'OR': ['aqf_system_all'], 'AND': []}
    if settings.get('system_type'):
        # Include tests for this system type.
        condition['OR'].append("aqf_system_%s" % settings['system_type'])

    if settings.get('system_location', '').startswith('karoo'):
        # Disable intrusive if in the Karoo.
        condition['AND'].append("(not aqf_intrusive)")
    else:
        # Disable site_only if not in karoo.
        condition['AND'].append("(not aqf_site_only)")

    #if settings.get('site_acceptance'):
    #    if settings.get('system_location', '').startswith('karoo'):
    #        # Include site_only tests if in the Karoo
    #        condition['AND'].append("(aqf_site_acceptance or aqf_site_only)")
    #    else:
    #        condition['AND'].append("aqf_site_acceptance")

    # Set the include/exclude criteria for Qualification/Acceptance Testing/Demonstration
    if settings.get('demo'):
        if settings.get('site_acceptance'):
            # For Acceptance Demonstration:
            # run aqf_demo_tests decorated with site_acceptance
            # and aqf_site_tests
            condition['AND'].append('(aqf_site_test or (aqf_demo_test and aqf_site_acceptance))')
        else:
            # For Qualification Demonstration:
            # run aqf_demo_tests
            condition['AND'].append('aqf_demo_test')
    else: # Not demo mode - thus this is AUTO testing
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

    # Mix OR with AND
    condition['or_str'] = ' or '.join(condition['OR'])
    if condition['AND'] and condition['or_str']:
        condition['AND'].append("(%s)" % condition['or_str'])
        cmd.append('-A(%s)' % ' and '.join(condition['AND']))
    elif condition['AND']:
        cmd.append('-A(%s)' % ' and '.join(condition['AND']))
    elif condition['or_str']:
        cmd.append('-A(%s)' % condition['or_str'])

    katreport_control = []
    if settings.get('jenkins'):
        katreport_control.append('jenkins')
    if settings.get('demo'):
        katreport_control.append('demo')
    if settings.get('katreport_quick'):
        katreport_control.append('quick')
    if katreport_control:
        cmd.append("--katreport-control=%s" % ','.join(katreport_control))

    if settings.get('site_acceptance'):
        # Use different directory for acceptance results, so as not to overwrite qualification results
        cmd.append("--katreport-name=katreport_accept") # Using default "katreport" for qualification

    if settings.get('jenkins'):
        cmd.append("--with-xunit")
        cmd.append("--xunit-file=build/nosetests.xml")
    cmd.extend(settings.get("tests") or ['./tests'])

    nose_args = settings.get('nose_args', '')
    for arg in nose_args.split():
        cmd.append(arg)
    # Run with --logging-level WARN if logging-level not passed in with nose_args
    if "logging-level" not in nose_args:
        cmd.append("--logging-level=WARN")

    # Let the output log be written into the katreport_dir
    cmd.append(" 2>&1 | tee %s/output.log" % (katreport_dir))

    run_command(settings, log_func, cmd, shell=True)


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
    files = {'test': os.path.join(settings['me_dir'],
                                  '{}/katreport.json'.format(katreport_dir)),
             'system': os.path.join(settings['me_dir'],
                                    '{}/katreport_system.json'.format(katreport_dir)),
             'core': settings.get('json_file', '')}
    return files.get(what, None)

def generate_report(settings, log_func):
    report_type = settings['report'].lower()
    katreport_dir = settings["katreport_dir"]
    files = {'test': get_filename('test', settings),
             'system': get_filename('system', settings),
             'core': get_filename('core', settings)}
    urls_for_jenkins = {'test': "katreport.json",
                        'system': "katreport_system.json"}

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

    from report_generator.report import Report

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
    log_func('INFO', 'Remove HTML files and results from previous tests from {}'.format(katreport_dir))
    os.chdir(settings.get('me_dir'))
    run_command(settings, log_func, ['make', 'clean'])
    run_command(settings, log_func, ['rm', '-rf', katreport_dir])
    try:
        os.makedirs(katreport_dir)
    except os.error:
        pass


def gather_system_settings(settings, log_func):
    """Get information of the system we are running on."""
    katreport_dir = settings["katreport_dir"]
    filename = os.path.join(settings['me_dir'], katreport_dir,
                            'katreport_system.json')
    data = {'pip_version': {}, 'dpkg_version': {}, 'vcs_version': {}}

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

    cmd = ['kat-versioncontrol.py']
    for line in str.splitlines(run_command_output(settings, log_func, cmd)):
        segments = line.split(":")
        if len(segments) > 1:
            data['vcs_version'][segments[0]] = segments[1]
    data['dist'] = platform.dist()
    data['uname'] = os.uname()
    data['environment'] = os.environ
    data['environment'] = dict([(i, str(data['environment'].get(i, '')))
                                for i in data['environment']])
    data['ip_addresses'] = str.splitlines(
        run_command_output(settings, log_func, ['ip', '-f', 'inet', 'addr']))
    # Here we define the labels that katreport will use. Dont want katreport
    # to have to much knowledge of this file.
    data['Labels'] = {'pip_version': {'label': 'Python Modules',
                                      'description': 'Versions of python '
                                      'modules managed by pip. This data was '
                                      'obtained by running "pip freeze."'},
                      'vcs_version': {'label': 'Versions from VCS',
                                      'description': 'Output from '
                                                     'kat-versioncontrol.py'},
                      'dpkg_version': {'label': 'Installed Software',
                                       'description': 'Versions of software '
                                       'installed on this system.'},
                      'dist': {'label': 'OS Distribution',
                               'description': 'Distribution of the '
                               'Operating System'},
                      'uname': {'label': 'Uname',
                                'description': 'Output of the uname command'},
                      'ip_addresses': {'label': 'IP Adresses',
                                       'description': 'IP Adresses of the '
                                       'system test was ran on'}}
    with open(filename, 'w') as fh:
        fh.write(json.dumps(data, indent=4))

if __name__ == "__main__":
    options, args = option_parser()
    settings = dict((k, getattr(options, k)) for k in dir(options)
                    if not callable(getattr(options, k))
                    and not k.startswith('_'))
    settings.update(get_system_info())
    if settings['manual_systype']:
        settings['systype'] = settings['manual_systype']
    settings['process_core'] = True
    settings['gather_system_settings'] = True
    settings['tests'] = args
    settings['me'] = os.path.abspath(__file__)
    settings['me_dir'] = os.path.dirname(settings['me'])
    if settings["site_acceptance"]:
        settings['katreport_dir'] = "katreport_accept"
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
    if settings.get('demo'):
        settings['report'] = 'skip'
        settings['gather_system_settings'] = False
    if settings.get('dry_run'):
        settings['report'] = 'skip'
    if settings.get('jenkins'):
        settings['dev_update'] = True
        settings['cleanup'] = True
        settings['slow_test'] = True
        settings['report'] = 'skip'
    if settings['report'] in ['jenkins']:
        settings['gather_system_settings'] = False
    if settings['report'] in ['skip']:
        settings['process_core'] = False
        settings['gen_html'] = False

    # Do the different steps.
    log_func = create_log_func(settings)
    if settings.get('dev_update'):
        do_dev_update(settings, log_func)
    if settings.get('cleanup'):
        do_cleanup(settings, log_func)
    if settings['gather_system_settings']:
        gather_system_settings(settings, log_func)
    if settings['process_core']:
        process_core_data(settings, log_func)
    now = time.localtime()
    start_time = ("%02d%02d%02d-%02dh%02d" %
                     (now.tm_year, now.tm_mon, now.tm_mday,
                      now.tm_hour, now.tm_min))
    settings['build_dir'] = "build-"+start_time
    if settings['verbose']:
        print "=========settings========="
        for key in settings:
            print key,":",settings[key]
        print "=========================="
    if (settings['report'] in ['local_&_test', 'skip']
            or settings.get('dry_run')):
        run_nose_test(settings, log_func)
    if settings['report'] in ['results']:
        show_test_results(settings, log_func)
    elif settings['report'] not in ['skip']:
        generate_report(settings, log_func)
        if settings['gen_html']:
            generate_html_sphinx_docs(settings, log_func)

#os.system('rm *.svg')
