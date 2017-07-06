from __future__ import with_statement

import datetime
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import traceback

import colors
import numpy as np
from nose.plugins import Plugin

log = logging.getLogger('nose.plugins.nosekatreport')
test_logger = logging.getLogger('mkat_fpga_tests')

try:
    import matplotlib.pyplot
    matplotlib.use('Agg', warn=False, force=True)
except ImportError:
    log.info('Matplotlib not found, will not be able to add matplotlib figures')

# try:
#    ReSTProducer
# except NameError:
#    from .rest_producer import ReStProducer
# from .report import Report

__all__ = ['KatReportPlugin', 'Aqf', 'StoreTestRun']

UNKNOWN = 'unknown'
WAIVED = 'waived'
PASS = 'passed'
ERROR = 'error'
FAIL = 'failed'
TBD = 'tbd'
SKIP = 'skipped'


class TestFailed(AssertionError):
    """Raise AssertionError when a test fails"""
    pass


class StoreTestRun(object):
    """Class to store the state of the running test.

    Provide some additional helper functions for output formating.

    """

    def __init__(self):
        self.test_run_data = {'Meta': {'start': str(datetime.datetime.utcnow()),
                                       'end': None,
                                       'sys_args': sys.argv}}
        self.test_name = 'Unknown'
        self.step_counter = 0
        self.progress_counter = 0
        # Is re-initialised by add_test(), but also here to allow tests to run when the
        # plugin is not active in nosetests.
        self.test_image_counter = 0
        self.requirements_file = None
        self.test_passed = True
        self.test_failed = False
        self.test_skipped = False
        self.test_tbd = False
        self.test_waived = False
        self.test_ack = False
        self.error_msg = ''

        # Get the sitename. Hacky, should cleanup a bit.
        if os.path.isfile('/var/kat/node.conf'):
            with open('/var/kat/node.conf', 'r') as fh:
                node_conf = json.loads(fh.read())
            self.test_run_data['Meta']['sitename'] = '{0}_{1}'.format(
                node_conf.get('site'), node_conf.get('systype'))
        filename = '/var/kat/sitename'
        if os.path.isfile(filename):
            with open(filename, 'r') as fh:
                self.test_run_data['Meta']['sitename'] = fh.read().strip()
        self.image_tempdir = tempfile.mkdtemp()

    def add_test(self, test_name, **kwargs):
        """Add a new tests to the documentation."""
        Aqf.log_test(test_name)
        self.test_passed = True  # Be optimistic.
        self.test_skipped = False
        self.test_tbd = False
        self.test_waived = False
        self.test_ack = False
        self.error_msg = ''
        self.test_name = test_name
        # Re-initialise test_image_counter to zero for each test
        self.test_image_counter = 0
        labels = test_name.split(".")
        test_label = str(labels.pop()).replace("_", ' ').title()
        try:
            for _ in xrange(1, 4):
                group_label = labels.pop()
        except IndexError:
            pass
        group_label = group_label.replace("_", ' ').title()
        self._update_test(self.test_name,
                          {'steps': {}, 'label': test_label,
                           'group': group_label, 'demo': False})
        self.step_counter = 0
        # Cleanup the requirements.
        if 'requirements' in kwargs:
            if isinstance(kwargs['requirements'], list):
                kwargs['requirements'] = list(set(
                    [str(val).strip()
                     for val in kwargs['requirements'] if val]))
            Aqf.log_line(kwargs['requirements'])
        # Cleanup the description.
        if kwargs.get('description'):
            kwargs['description'] = str(kwargs['description']).strip()
        self._update_test(test_name, kwargs)
        Aqf.log_line("=" * 80)  # Separation line

    def add_step(self, message=None, hop=False):
        """Add a step to a test."""
        if self.step_counter > 0:
            # Record the end of the previous step
            self._update_step({'_updated': True},
                              {'type': 'CONTROL', 'msg': 'end'})
        self.step_counter += 1
        # Record the start of the next step
        step_data = {'status': PASS, 'success': True,
                     'description': message,
                     'step_start': str(datetime.datetime.utcnow()),
                     'progress': [], 'evaluation': [], }
        step_data['hop'] = hop
        step_action = {'type': 'control', 'msg': 'start'}
        self._update_step(step_data, step_action)

    def add_step_evaluation(self, description):
        """Add evaluation information to the test."""
        self._update_step({'_updated': True},
                          {'type': 'evaluate', 'msg': description})

    def add_step_checkbox(self, description, status):
        """
        Add a checkbox step that will produce PASSED/FAILED
        and Press any key to continue
        """
        self._update_step({'_updated': True, 'status': status},
                          {'type': 'checkbox', 'msg': description})

    def add_step_keywait(self, description, status):
        """
        Add a keywait step that will wait for a key
        before continuing but not print PASSED/FAILED as does checkbox
        """
        self._update_step({'_updated': True, 'status': status},
                          {'type': 'keywait', 'msg': description})

    def add_step_waived(self, description):
        """Add evaluation information to the test."""
        self._update_step({'_updated': True},
                          {'type': 'waived', 'msg': description})

    def add_progress(self, message):
        """Add progress information to a step."""

        # This is not used any more Aqf.progress logs to stdout only.
        self._update_step({'_updated': True},
                          {'type': 'progress', 'msg': message})

    def add_image(self, filename, caption="", alt=""):
        """Add an image to the report

        Parameters
        ----------
        filename : str
            Name of the image file.
        caption : str
            Caption text to go with the image
        alt : str
            Alternate text in case the image is not rendered

        Note a copy of the file will be made, and the test name as well as step number
        will be prepended to filename to ensure that it is unique

        """
        base_filename = os.path.basename(filename)
        prepended_filename = "{:04d}_{}_{:03d}_{}".format(
            self.step_counter, self.test_name, self.test_image_counter, base_filename)
        self.test_image_counter += 1
        try:
            shutil.copy(filename, os.path.join(self.image_tempdir, prepended_filename))
        except IOError:
            log.error('Failed to copy filename:%s to %s' % (filename, self.image_tempdir))
        else:
            final_filename = os.path.join('images', prepended_filename)
            self._update_step({'_updated': True}, dict(type='image', filename=final_filename,
                                                       caption=caption, alt=alt))

    def add_matplotlib_fig(self, filename, caption="", alt="", autoscale=False):

        """Save current matplotlib figure to the report

        Parameters
        ----------
        filename : str
            The filename to which to save the figure. Extension determines type, as
            supported by matplotlib
        caption : str, optional
            Caption for use in the report.
        alt : str, optional
            Alternative description for when an image cannot be displayed

        """
        if autoscale:
            matplotlib.pyplot.autoscale(tight=True)
            try:
                matplotlib.pyplot.tight_layout()
            except ValueError:
                pass
        try:
            matplotlib.pyplot.savefig(filename, bbox_inches='tight', dpi=200, format='png')
        except Exception:
            pass
        else:
            try:
                matplotlib.pyplot.cla()
            except Exception:
                matplotlib.pyplot.clf()
            self.add_image(filename, caption, alt)
            # matplotlib.pyplot.close('all')

    def as_json(self):
        """Output report in json format.

        :return: String. Json Data

        """
        self.test_run_data['Meta']['end'] = str(datetime.datetime.utcnow())
        return json.dumps(self.test_run_data, sort_keys=True, indent=4)

    def set_step_state(self, state, message=None):
        """Set the state of the step."""
        log_func = getattr(Aqf, "log_%s" % state)
        log_func(message)
        action = {'type': state, 'msg': message}
        if state in [PASS, WAIVED, SKIP, TBD]:
            # Record step as a success
            step_success = True
        else:  # UNKNOWN, ERROR, FAIL
            # Record step as a failure
            # stack = traceback.format_stack()
            # action['stack'] = stack
            step_success = False
            # Aqf.log_traceback('Last 6 lines of stack:\n' +
            #                  ' '.join(stack[-7:-1]))

        self._update_step({'status': state, 'success': step_success,
                           'error_msg': message}, action)

    def set_test_state(self, test_name, state, err_obj=None):
        if state in [PASS, WAIVED]:
            # Record test as a success
            test_success = True
        else:
            # Record test as a failure
            test_success = False
        data = {'status': state, 'success': test_success}
        if err_obj:
            data['err_obj'] = str(err_obj)
            data['error_msg'] = str(err_obj[1])
            data['tb'] = str(err_obj[2])
            if hasattr(err_obj[2], 'format_tb'):
                data['traceback'] = err_obj[2].format_tb()
                # data['stack'] = err_obj[2].format_stack()

        self._update_test(test_name, data)

    def read_requirements(self, requirements_file):
        """Read requirements from a JSON file."""
        if requirements_file and os.path.isfile(requirements_file):
            self.requirements_file = requirements_file

    def write_test_results_json_file(self, filename, clean_tempdir=True):
        """Write the stored data as a JSON file.

        Will also move report images into an 'images' subdir in the same directory
        as the JSON file.

        Will clean out the image tempdir if clean_tempdir=True

        """
        with open(filename, 'w') as fh:
            fh.write(self.as_json())
        self._copy_and_clean_image_files(filename, clean_tempdir)

    def _copy_and_clean_image_files(self, json_filename, clean_tempdir=True):
        report_dir = os.path.dirname(json_filename)
        images_dir = os.path.join(report_dir, 'images')
        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)
        shutil.copytree(self.image_tempdir, images_dir)

        # Since images are created in a tempdir the permissions may be too restrictive, so
        # let's have a look at the json file's permissions and set group / other read
        # rights as needed
        add_perms = []
        json_stat = os.stat(json_filename)
        if json_stat[stat.ST_MODE] & stat.S_IRGRP:
            add_perms.append('g+r')
            add_perms.append('g+X')

        if json_stat[stat.ST_MODE] & stat.S_IROTH:
            add_perms.append('o+r')
            add_perms.append('o+X')

        subprocess.check_call(['chmod', '-R', ','.join(add_perms), images_dir])
        if clean_tempdir:
            shutil.rmtree(self.image_tempdir)

    def _comp_status(self, status1, status2):
        """Return the more critical ststus."""
        # Rolled up VR status of PASS/ SKIPPED /WAIVED must be checked: order
        # is FAIL/ERROR, TBD, SKIPPED, PASSED, WAIVED (Thus anything for which one
        # step SKIP results in SKIP, anything of which all is PASSED but one is
        # WAIVED results in PASSED)
        status = [UNKNOWN,  # Dont know what happened
                  PASS,  # Test Passed
                  WAIVED,  # The test was waived.
                  SKIP,  # Skip this test
                  TBD,  # Test is to-be-done
                  FAIL,  # Test Failed
                  ERROR]  # Something went wrong
        try:
            st1 = status.index(status1)
        except ValueError:
            st1 = 0
        try:
            st2 = status.index(status2)
        except ValueError:
            st2 = 0
        return status[max(st1, st2)]

    def _update_step(self, data, action=None):
        """Update the information of a step."""
        if action is None:
            action = {}
        if self.test_name not in self.test_run_data:
            self.test_run_data[self.test_name] = {'steps': {}}
        elif 'steps' not in self.test_run_data[self.test_name]:
            self.test_run_data[self.test_name]['steps'] = {}
        __ = self.test_run_data[self.test_name]['steps']
        # __ is the internal reference to the steps list in a step.
        if not __.get(self.step_counter):
            __[self.step_counter] = data
        else:
            data['success'] = all([__[self.step_counter].get('success', True),
                                   data.get('success', True)])
            if 'status' in data:
                new_status = self._comp_status(data['status'],
                                               __[self.step_counter].get('status', PASS))

                if new_status != data.get('status'):
                    data['status'] = new_status
                self._update_test(self.test_name,
                                  {'status': data.get('status'),
                                   'success': data.get('success', True),
                                   'error_msg': data.get('error_msg', '')})
            __[self.step_counter].update(data)

        # ACTION
        if action:
            if 'action' not in __[self.step_counter]:
                __[self.step_counter]['action'] = []

            action['time'] = str(datetime.datetime.utcnow())
            __[self.step_counter]['action'].append(action)

    def _update_test(self, test_name, data):
        """Update the information of a test."""
        if not self.test_run_data.get(test_name):
            self.test_run_data[test_name] = {}
        if 'success' in data:
            data['success'] = all([data['success'],
                                   self.test_run_data[test_name].get('success', True)])
            self.test_passed = data['success']
        if 'status' in data:
            data['status'] = self._comp_status(
                data['status'],
                self.test_run_data[test_name].get('status', PASS))
            self.test_skipped = data['status'] == SKIP
            self.test_tbd = data['status'] == TBD
            self.test_waived = data['status'] == WAIVED
            if (self.test_run_data[test_name].get('status') !=
                    data.get('status')):
                # status degraded. Store the message
                self.error_msg = data.get('error_msg')
        self.test_run_data[test_name].update(data)


class _state(object):
    """Class for storing state and progress."""

    report_name = 'katreport'  # dir name that reports are writen into
    store = StoreTestRun()
    global_systems = {}  # Track the systems that a test object was marked with
    config = {}


class KatReportPlugin(Plugin):
    name = 'katreport'
    config = {'demo': False}

    def options(self, parser, env=os.environ):
        super(KatReportPlugin, self).options(parser, env=env)
        parser.add_option('--katreport-name', action='store',
                          default='katreport',
                          dest='katreport_name',
                          help='Name of the directory to generate; '
                               ' defaults to katreport. Note that this'
                               ' directory is overwritten')
        parser.add_option('--katreport-requirements', action='store',
                          dest='requirements_file', metavar="FILE",
                          help='Name of file that has the requirements')
        parser.add_option('--katreport-control', action='store',
                          dest='control_string',
                          help='Comma separated list of control keywords')

    def configure(self, options, conf):
        super(KatReportPlugin, self).configure(options, conf)
        if not self.enabled:
            return
        if not os.path.exists(options.katreport_name):
            os.makedirs(options.katreport_name)
        _state.report_name = options.katreport_name
        _state.store.read_requirements(options.requirements_file)
        if options.control_string:
            control = str(options.control_string).lower().split(',')
            for cntr in control:
                if ":" in cntr:
                    split_cntr = cntr.split(":")
                    self.config[split_cntr[0]] = split_cntr[1]
                else:
                    self.config[cntr] = True
        _state.config = self.config

    def addFailure(self, test, err):
        _state.store.set_test_state(test.id(), FAIL, err)

    def addError(self, test, err):
        _state.store.set_test_state(test.id(), ERROR, err)

    def addSuccess(self, test):
        _state.store.set_test_state(test.id(), PASS)

    def addSkip(self, test):
        _state.store.set_test_state(test.id(), SKIP)

    def addTbd(self, test):
        _state.store.set_test_state(test.id(), TBD)

    def startContext(self, context):
        _state.global_systems = {}
        for system in [n for n in dir(context)
                       if n.startswith("aqf_system_")]:
            sys_name = system.replace("aqf_system_", "").upper()
            _state.global_systems[sys_name] = all([getattr(context, system)])

    def stopTest(self, nose_test=None):
        test = nose_test.test
        # _state.store.add_progress("END")
        mesg = ("%s\n\t\tDid not reach the end of the test. Either the test "
                "did not end with Aqf.end() or there was a critical error." %
                test.id())
        if not _state.store.test_ack:
            sys.stderr.write("\n%s\n" % mesg)
            if _state.store.test_skipped or _state.store.test_tbd or _state.store.test_waived:
                import nose
                raise nose.plugins.skip.SkipTest
            else:
                amsg = ("\n\t\t%s\n\t\tTest failed because not all steps "
                        "passed\n\t\t%s\n\t\t%s" %
                        (mesg, _state.store.test_name, _state.store.error_msg))
                assert _state.store.test_passed, amsg

    def startTest(self, nose_test):
        test = nose_test.test
        test_method = getattr(test, test._testMethodName)

        try:
            requirements = test_method.katreport_requirements
        except AttributeError:
            requirements = []

        aqf_attr = {'systems': _state.global_systems}
        for attr in [n for n in dir(test_method)
                     if n.startswith("aqf_")]:
            if attr.startswith('aqf_system_'):
                aqf_attr['systems'][attr.replace("aqf_system_", "").upper()] = all(
                    [getattr(test_method, attr)])
            else:
                aqf_attr[attr] = getattr(test_method, attr)

        _state.store.add_test(test.id(),
                              description=test_method.__doc__,
                              requirements=requirements,
                              **aqf_attr)

    def finalize(self, result):

        # Set the end time in the Json file.
        _state.store.test_run_data['Meta']["end"] = str(datetime.datetime.utcnow())

        # Write the test results to the JSON file.
        _state.store.write_test_results_json_file(os.path.join(_state.report_name,
                                                               'katreport.json'))


class AqfLog(type):
    """
    Catch all the method calls that start with log_.

    This is a helper class for Aqf, all Aqf methods are classmethods so unable
    to use __getattr__ on them. This is a little sneaky, but works well.

    """

    def __getattr__(cls, name):
        if name.startswith("log_"):
            def func(*arg):
                for line in arg:
                    strip_name = name.replace("log_", "")
                    cls._log_msg(strip_name, str(line))

            return func

    def _severity_colour(self, severity):
        """
        'black', 'blink', 'blink2', 'blue', 'bold', 'concealed', 'crossed',
        'cyan', 'faint', 'green', 'italic', 'magenta', 'negative'
        'red', 'underline', 'white', 'yellow'

        :return: String(10). Severity with colour formatting.

        """
        if _state.config.get('jenkins'):
            return "{:10s}".format(severity)
        sc = {'INFO': colors.blue,
              'ERROR': colors.red,
              'TRACEBACK': colors.red,
              'EXIT': colors.red,
              'PASSED': colors.cyan,
              'WAIT': colors.faint,
              'TEST': colors.negative,  # colors.blink
              'STEP': colors.bold,
              'TBD': colors.blue,
              'SKIPPED': colors.blue,
              'CHECKBOX': colors.magenta,
              'KEYWAIT': colors.magenta,
              'BUILD': colors.faint,
              'FAILED': colors.yellow,
              'PROGRESS': colors.faint,
              'HOP': colors.faint,
              'WAIVED': colors.underline
              }
        call_col = sc.get(severity, None)
        if call_col:
            fmt_severity = sc[severity]("{:10s}".format(severity))
        elif severity == "LINE":
            fmt_severity = "{:10s}".format(" ")
        else:
            fmt_severity = "{:10s}".format(severity)
        return fmt_severity

    def _log_msg(self, severity, message):
        """Print direct to standard error, nose will not catch the message."""
        time = datetime.datetime.strftime(datetime.datetime.utcnow(),
                                          '%H:%M:%S.%f')
        severity = severity.upper()
        if severity == "TEST":
            # Add a separation line before each test as PROGRESS
            sys.stderr.write("\n{} {}: {}\n".format
                             (time, self._severity_colour('TEST'), "=" * 80))
        if _state.config.get('demo'):
            # Also print BUILD and HOP in DEMO mode
            ###if severity in ['BUILD', 'HOP']:
            ###    return
            sys.stderr.write("{} {}: {}\n".format
                             (time, self._severity_colour(severity), message))
        else:
            sys.stderr.write("{} {}: {}\n".format
                             (time, self._severity_colour(severity), message))


class Aqf(object):
    """Automatic Qualification Framework.

    The AQF class is used as a container to the public class methods.
    The AQF should not be instantiated.

    Supports logging to the console.
    The methods Aqf.log_debug(msg) is available for debugging and writing test.
    These log messages will not be recorded or filtered by nose.

    """

    # We are calling classmethods so we need to be extra sneaky. when we define
    # a generic handler for the log messages.
    __metaclass__ = AqfLog

    @classmethod
    def wait(cls, seconds, message=None):
        """Wait for seconds.

        A progress message is added to the report.
        :param seconds: Int. Seconds to sleep.

        """
        if message:
            cls.log_wait("%s - %.3f seconds" % (message, float(seconds)))
        else:
            cls.log_wait("%.3f seconds" % (float(seconds)))
        time.sleep(seconds)

    @classmethod
    def progress(cls, message):
        """Add progress messages to the step."""
        # _state.store.add_progress(message)
        _state.store.add_progress(message)
        cls.log_progress(message)
        # test_logger.info(message)


    @classmethod
    def hop(cls, message=None):
        """A internal step in a test section.

        Hop is like a step but hop is used for internal setup of the test
        environment.
        eg. Aqf.hop("Test that the antenna is stowed when X is set to Y")

        :param message: String. Message describe what the hop will test.

        """
        if message is None:
            message = "Doing Setup"
        _state.store.add_step(message, hop=True)
        cls.log_hop(message)
        test_logger.info(message)

    @classmethod
    def step(cls, message):
        """A step in a test section.

        eg. Aqf.step("Test that the antenna is stowed when X is set to Y")

        :param message: String. Message describe what the step will test.

        """
        _state.store.add_step(message)
        cls.log_step(message)
        # test_logger.info(message)

    @classmethod
    def stepBold(cls, message):
        """A step bolded in a test section.

        eg. Aqf.stepBold("Test that the antenna is stowed when X is set to Y")

        :param message: String. Message describe what the step will test.

        """
        try:
            assert isinstance(message, list)
        except AssertionError:
            message = [message]
        message = colors.bold('{:10s}'.format(''.join(message)))
        _state.store.add_step(message)
        cls.log_step(message)

    @classmethod
    def stepline(cls, message):
        """A step underlined in a test section.

        eg. Aqf.stepline("Test that the antenna is stowed when X is set to Y")

        :param message: String. Message describe what the step will test.

        """
        try:
            assert isinstance(message, list)
        except AssertionError:
            message = [message]
        message = colors.underline(''.join(message))
        _state.store.add_step(message)
        cls.log_step(message)

    @classmethod
    def addLine(cls, linetype, count=80):
        """A step in a test section with lines

        eg. Aqf.step("Test that the antenna is stowed when X is set to Y")

        :param linetype: String. linetype eg: * - _
        :param count: Int. How long do you want your line to be.

        """
        message = linetype * count
        _state.store.add_step(message, hop=True)
        cls.log_step(message)
        test_logger.info(message)

    @classmethod
    def image(cls, filename, caption='', alt=''):
        _state.store.add_image(filename, caption, alt)

    @classmethod
    def matplotlib_fig(self, filename, caption="", alt="", autoscale=False):
        """Save current matplotlib figure to the report

        Parameters
        ----------
        filename : str
            The filename to which to save the figure. Extension determines type, as
            supported by matplotlib
        caption : str, optional
            Caption for use in the report.
        alt : str, optional
            Alternative description for when an image cannot be displayed

        """
        _state.store.add_matplotlib_fig(filename, caption, alt, autoscale)

    @classmethod
    def substep(cls, message):
        """A sub step of a step in a test section.

        eg. Aqf.substep("Test that the antenna no 3 is stowed")

        :param message: String. Message describe what the sub step will test.

        """
        _state.store.add_step(message)
        cls.log_substep(message)

    @classmethod
    def passed(cls, message=None):
        """Test step Passed.

        The step passed.

        eg. Aqf.passed()

        :param message: Optional String. Message to add to the test step.

        """
        _state.store.set_step_state(PASS, message)

    @classmethod
    def failed(cls, message=None):
        """Test section failed.

        The step failed.

        eg. Aqf.failed("Antenna could not be placed in stow position.")

        :param message: Optional String. Reason test step failed.

        """
        _state.store.set_step_state(FAIL, message)

    @classmethod
    def skipped(cls, message=None):
        """Test step is skipped.

        The test did not pass or fail, but for some reason could not be
        executed at this time. This will not mark the requirement as
        fulfilled.

        eg. Aqf.skipped("Antenna is already in stow position.")

        :param message: Optional String. Reason for skipping the test step.

        """
        _state.store.set_step_state(SKIP, message)

    @classmethod
    def tbd(cls, message=None):
        """Test step is to-be-done - handled same as SKIPPED.

        The test did not pass or fail, but are not executed at this time as it is
        still to-be-done. This will not mark the requirement as
        fulfilled.

        eg. Aqf.tbd("This test is still to-be-done.")

        :param message: Optional String. Reason for the test step to be TBD.

        """
        _state.store.set_step_state(TBD, message)

    @classmethod
    def waived(cls, message):
        """Test step is waived.

        This test has been waived for a reason.
        eg. Aqf.waived("This requirement is postponed till Timescale C.")

        :param message: String. Reason for skipping the test step.

        """
        _state.store.add_step_waived(message)
        #_state.store.set_step_state(WAIVED, message)

    @classmethod
    def equals(cls, result, expected, description):
        """Evaluate: expected result equals the obtained result.

        Shortcut for: ::

            if expected == result:
                Aqf.pass()
            else:
                Aqf.failed(message)

        :param result: The obtained value.
        :param expected: The expected value.
        :param description: A description of this test.

        """

        # Do not log EVALUATE step
        # _state.store.add_step_evaluation(description)
        if expected == result:
            cls.passed(description)
            return True
        else:
            cls.failed("Expected '%s' got '%s' - %s" %
                       (str(expected), str(result), description))
            return False

    @classmethod
    def is_not_equals(cls, result, expected, description):
        """Evaluate: expected result is not equals the obtained result.

        Shortcut for: ::

            if expected != result:
                Aqf.pass()
            else:
                Aqf.failed(message)

        :param result: The obtained value.
        :param expected: The expected value.
        :param description: A description of this test.

        """

        # Do not log EVALUATE step
        # _state.store.add_step_evaluation(description)
        if expected != result:
            cls.passed(description)
        else:
            cls.failed("Expected '%s' got '%s' - %s" %
                       (str(expected), str(result), description))

    @classmethod
    def array_abs_error(cls, result, expected, description, abs_error=0.1):
        """
        Compares absolute error in numeric result and logs to Aqf.

        Parameters
        ----------
        result: numeric type or array of type
            Actual result to be checked.
        expected: Same as result
            Expected result
        description: String
            Message describing the purpose of the comparison.
        abs_err: float, optional
            Fail if absolute error is not less than this abs_err for all array
            elements

        """
        err = np.abs(np.array(expected) - np.array(result))
        max_err_ind = np.argmax(err)
        max_err = err[max_err_ind]
        if max_err >= abs_error:
            cls.failed('Absolute error larger than {abs_error}, max error at index {max_err_ind}, '
                       'error: {max_err} - {description}'.format(**locals()))
            return False
        else:
            cls.passed(description)
            return True

    @classmethod
    def array_almost_equal(cls, result, expected, description, **kwargs):
        """Compares numerical result to an expected value

        Using numpy.testing.assert_almost_equal for the comparison

        Parameters
        ----------
        result: numeric type or array of type
            Actual result to be checked.
        expected: Same as result
            Expected result
        description: String
            Message describing the purpose of the comparison.
        **kwargs : keyword arguments
            Passed on to numpy.testing.assert_almost_equal. You probably want to use the
            `decimal` kwarg to specify how many digits after the decimal point is compared.
        """
        try:
            np.testing.assert_almost_equal(result, expected, **kwargs)
        except AssertionError:
            cls.failed("Expected '%s' got '%s' - %s" %
                       (str(expected), str(result), description))
        else:
            cls.passed(description)

    @classmethod
    def in_range(cls, result, expected_min, expected_max, description):
        """Evaluates: obtained result to a minimum and maximum expected value (interval comparison)

        Parameters
        ----------
        result: numeric type
            Actual result to be checked.
        expected minimum: numeric type
            Expected minimum result to be checked against actual results
        expected maximum: numeric type
            Expected maximum result to be checked against actual results
        description: String
            Message describing the purpose of the comparison.
        """
        try:
            assert expected_min <= result <= expected_max
        except AssertionError:
            cls.failed("Actual value '%s' is not between '%s' and '%s' -  %s" % (
                str(result), str(expected_min), str(expected_max), description))
        else:
            cls.passed(description)

    @classmethod
    def less(cls, result, expected, description):
        """Evaluate: obtained result less than the expected value.

        Shortcut for: ::

            if result < expected:
                Aqf.pass()
            else:
                Aqf.failed('Result: {result} not less than {expected}')

        :param result: The obtained value.
        :param expected: The expected value.
        :param description: A description of this test.

        """

        if result <= expected:
            cls.passed(description)
            return True
        else:
            cls.failed('Result {result} not less than {expected} - {description}'
                       .format(**locals()))
            return False

    @classmethod
    def more(cls, result, expected, description):
        """Evaluate: obtained result more than the expected value.

        Shortcut for: ::

            if result > expected:
                Aqf.pass()
            else:
                Aqf.failed('Result: {result} not less than {expected}')

        :param result: The obtained value.
        :param expected: The expected value.
        :param description: A description of this test.

        """

        if result >= expected:
            cls.passed(description)
            return True
        else:
            cls.failed('Result {result} not less than {expected} - {description}'
                       .format(**locals()))
            return False

    @classmethod
    def almost_equals(cls, result, expected, tolerance, description):
        """Evaluate: expected result equals the obtained result within the tolerance.
        Meaningful only for numeric values.

        Shortcut for: ::

            if abs(expected-result) <= tolerance:
                Aqf.pass()
            else:
                Aqf.failed(message)

        :param result: The obtained value.
        :param expected: The expected value.
        :param tolerance: The maximum absolute difference allowed between `expected` and `result`.
        :param description: A description of this test.

        """

        # Do not log EVALUATE step
        # _state.store.add_step_evaluation(description)
        if abs(expected - result) <= tolerance:
            cls.passed(description)
            return True
        else:
            cls.failed("Expected '%s' +/- '%s' got '%s' - %s" %
                       (str(expected), str(tolerance), str(result), description))
            return False

    @classmethod
    def is_true(cls, result, description):
        """Evaluate: expected result is true.

        Shortcut for: ::

            if result:
                Aqf.pass()
            else:
                Aqf.failed(message)

        Parameters
        ----------
        result: Boolean
            Value to be evaluated.
        description: String
            Message describing the purpose of the check.

        """
        if result:
            cls.equals(True, True, description)
            return True
        else:
            cls.equals(False, True, description)
            return False

    @classmethod
    def is_false(cls, result, description):
        """Evaluate: expected result is false.

        Shortcut for: ::

            if not result:
                Aqf.pass()
            else:
                Aqf.failed(message)

        Parameters
        ----------
        result: Boolean
            Value to be evaluated.
        description: String
            Message used when result is not false.

        """
        if not result:
            cls.equals(True, True, description)
            return True
        else:
            cls.equals(True, False, description)
            return False

    @classmethod
    def checkbox(cls, description):
        """Mark a step that should be manually confirmed."""
        # _state.store.mark_test_as_demo()
        cls.log_checkbox(description)
        if not _state.config.get('demo'):
            cls.log_checkbox("PASSED / FAILED")
            status = PASS
        else:
            cls.log_checkbox("PASSED / FAILED      <--- Press Any Key To Continue --->")
            key = wait_for_key(-1)  # No timeout
            sys.stderr.write("\r" + " " * 40 + "\n")
            if key is False:
                cls.log_checkbox("Continue from timeout")
                status = FAIL
            else:
                # cls.log_checkbox("Continue on key %s" % key)
                status = PASS
        _state.store.add_step_checkbox(description, status)

    @classmethod
    def keywait(cls, description):
        """
        Mark a step that should wait for a key before continuing.
        Don't print PASSED/FAILED
        """
        # _state.store.mark_test_as_demo()
        cls.log_keywait(description)
        if not _state.config.get('demo'):
            cls.log_keywait("Wait on keypress")
            status = PASS
        else:
            cls.log_keywait("                     <--- Press Any Key To Continue --->")
            key = wait_for_key(-1)  # No timeout
            sys.stderr.write("\r" + " " * 40 + "\n")
            if key is False:
                cls.log_keywait("Continue from timeout")
                status = FAIL
            else:
                # cls.log_keywait("Continue on key %s" % key)
                status = PASS
        _state.store.add_step_keywait(description, status)

    @classmethod
    def exit(cls, message=None):
        """Exit the system.

        This is not a nice thing to do so avoid it.

        """
        stack = traceback.format_stack()
        for line in stack:
            cls.log_exit(line.replace('\n', ' ').replace('\r', ' '))
        cls.log_exit(message)
        sys.stdout.flush()
        os._exit(1)

    @classmethod
    def end(cls, passed=None, message=None, traceback=None):
        """Mark the end of the test.

        Every test needs one of these at the end. This method will do the
        assert based on the status of the previous steps.

        :param passed: Optional Boolean. If the test passed of faile. If not
                       supplied the status of the step will be used.
        :param message: Optional string. Message to add to passed of failed.

        """
        if not traceback:
            sys.tracebacklimit = 0  # Disabled Traceback report

        if passed is True:
            cls.passed(message)
        elif passed is False:
            cls.failed(message)
            _state.store.test_failed = True  # , ("Test failed because not all steps passed\n\t\t%s\n\t\t%s" %
            # (_state.store.test_name, _state.store.error_msg))

        _state.store.test_ack = True
        _state.store._update_step({'_updated': True},
                                  {'type': 'CONTROL', 'msg': 'end'})
        if _state.store.test_skipped or _state.store.test_tbd or _state.store.test_waived:
            import nose
            raise nose.plugins.skip.SkipTest
        elif _state.store.test_failed:
            _state.store.test_failed = False
            fail_message = ("\n\nNot all test steps passed\n\t"
                                "Test Name: %s\n\t"
                                "Failure Message: %s\n"%(_state.store.test_name,
                                    _state.store.error_msg))
            fail_message = '\033[91m\033[1m %s \033[0m' %(fail_message)
            raise TestFailed(fail_message)
        else:
            try:
                assert _state.store.test_passed
            except AssertionError:
                fail_message = ("\n\nNot all test steps passed\n\t"
                                "Test Name: %s\n\t"
                                "Failure Message: %s\n"%(_state.store.test_name,
                                    _state.store.error_msg))
                fail_message = '\033[91m\033[1m %s \033[0m' %(fail_message)
                raise TestFailed(fail_message)

                # assert _state.store.test_passed, ("Test failed because not all steps passed\n\t\t%s\n\t\t%s" %
                # (_state.store.test_name, _state.store.error_msg))


def wait_for_key(timeout=-1):
    """Block until a key is pressed. Keys like shift and ctrl dont work.

    Adapted from: http://docs.python.org/2/faq/library
    :param timeout: Int. Seconds to wait for keypress. Default: -1 will wait forever.
    :returns: String or Boolean False. String value of key or False if timeout.

    """
    import termios
    import fcntl
    fd = sys.stdin.fileno()

    oldterm = termios.tcgetattr(fd)
    newattr = termios.tcgetattr(fd)
    newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSANOW, newattr)

    oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)
    try:
        timeout = int(timeout)
    except ValueError:
        timeout = 10
    start = time.time()
    result = None
    try:
        while result is None:
            try:
                # c = sys.stdin.read(1)
                c = sys.stdin.read()
                result = repr(c)
            except IOError:
                if timeout == -1:
                    # Contine to wait for a key
                    continue
                elif (time.time() - start) > timeout:
                    result = False
            time.sleep(1)
    finally:
        termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
        fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)

    return result
