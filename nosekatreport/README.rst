================
nose-kat-report
================

nose-kat-report is a nose_ plugin for writing an annotated test report
The test report generation is distributed throughout the tests through
calls to nose_kat_report

.. _nose: http://somethingaboutorange.com/mrl/projects/nose/


Installation
============

From mkat_fpga_tests/nosekatreport install nosekatreport package:

::

    sudo pip install . 


Invocation
==========

The simple way (from mkat_fpga_tests directory)::

  nosetests --with-katreport tests

DEPRECATED
==========

A utility script is provided to assist with running the test with the correct filters::

  ./run_aqf.py

The *run_aqf.py* script has several options. Please see *run_aqf.py --help* for up-to-date detail. ::

  ./run_aqf.py -h
    Usage: run_aqf.py [options] [tests]

    Options:
        -h, --help     show this help message and exit
        -v, --verbose  Be more verbose
        -q, --quiet    Be more quiet
        --acceptance   Will only run test marked '@site_acceptance' or linked
                       to .SITE. VRs, if in the Karoo then also @site_only
                       tests
        --demo         Run the tests linked to .DEMO. and .SITE. VRs in demo
                       mode. Wait for user input at each Aqf.checkbox
                       instance
        --dry_run      Do a dry run. Print commands that would be called by run_aqf
        --no_html      Do not generate the html output
        --quick        Only generate a small subset of the reports
        --no_slow      Exclude tests marked as @slow in this test run
        --clean        Cleanup reports from previous test run. Reports are replaced
                       by default without --clean. Clean is useful with --quick to
                       only generate the html of the test run report
       --report=REPORT       Only generate the reports. No tests will be run. Valid
                             options are: local_&_test (default), local, jenkins,
                             skip and results. 'results' will print the
                             katreport[_accept].json test results
       --nose          additional arguments to pass to nose e.g. --nose=--collect-only or --nose=--logging-level=DEBUG
       

Options can be mixed. Multiple tests can be given on the command line in a format that nose understands.
Tests can be granularly executed up to the test method. eg.::

    # <Path of test file>:<Object name>.<Method name>
    ./run_aqf.py tests/monitoring_logging_archiving/test_monitor.py:TestMonitor.test_device_and_proxy_down

    # OR

    # <Python import syntax>
    ./run_aqf.py tests.monitoring_logging_archiving.test_monitor.TestMonitor.test_device_and_proxy_down

Overview and Usage
============================

Typical AQF usage for Qualfication/Acceptance

*Update test decoratirs and CORE export:

  * Update CORE and export MeerKAT.xml, commit MeerKAT.xml to svn/katscripts/mkat_fpga_tests/supplemental
  * Update all the decorators and VR links (aqf_vr) in the tests.

* Perform the test runs:

  * Do a full qualfication and acceptance run: ::

  	python run_aqf.py ; python run_aqf.py --accept

  * This produces the Qualfication results and reports in svn/katscripts/mkat_fpga_tests/katreport/*.json
  * and the Acceptance results and reports in svn/katscripts/mkat_fpga_tests/katreport_accept/*.json
  * The test results land in katreport[_accept]/katreport.json
  * The reports produced are Qualification|Acceptance Testing|Demonstration Procedure|Results.

* Producing aqf_index.json for demo runs:
  * Once you are happy with the test results and all test decorators are done, then produce the aqf_index through: ::
  
  	python run_aqf.py --report=aqf_index

  * Copy katreport/aqf_index.json to svn/katscripts/mkat_fpga_tests/aqf_index.json and commit to SVN

* Perform the Demonstration event:
  * --rundemo=all,all   # to specify all timescales and all VRs
  * --rundemo=timescale,[start[,end]]   # to specify a timescale and optional start, end VR
  * python run_aqf.py --rundemo=all,VR.CM.DEMO.A.11,VR.CM.DEMO.CBF.55 # to specify start and end VR
  * python run_aqf.py --rundemo=all,VR.CM.DEMO.CBF.55 # to specify only start VR


Options
=======

General Options
---------------

``--katreport-option``
  Description of some option
  
``nosetests --help``

::

  --with-katreport      Enable plugin KatReportPlugin: (no help available)
                       [NOSE_WITH_KATREPORT]
  --katreport-name=KATREPORT_NAME
                       Name of the directory to generate;  defaults to
                       katreport. Note that this directory is overwritten
  --katreport-requirements=FILE



Docstring Markup
================

This plugin extracts docstrings from packages, modules, test classes
and test methods. If reStructuredText markup is used in docstings,
care should be taken that heading levels they define fit within the
report structure.

The Python documentation conventions_ are used for section headers.

.. _conventions: http://sphinx.pocoo.org/rest.html#sections


Report Generation
=================

The report is for the most part constructed from docstrings at the
test package, module, class and method level. Requirements can be
specified using the nosekatreport.satisfies_requirement
decorator. Sub-method granularly reporting can be done by the
Aqf.passed(`text`) function. The parameter `text` is
assumed to be the description of some sub-test that passed. Calls to
Aqf.passed() after a test assert that failed will not be added to
the report.


Decorators
==========

Requirements are added to a test with decorators, the reporting system will update the 
report with information from the CORE system based on the requirement given. Decorators are also used for filtering the tests so the correct test will run on the correct system.

The decorator can be used several times on the same method or function and
allows for several requirements to be given as arguments to the decorator.

eg. Using the decorator.::

    from nosekatreport import aqf_vr, system, Aqf, AqfBase

    @system('all')
    class TestAqf(AqfBase):

        @aqf_vr("VR.CM.AUTO.AB.12", "VR.CM.AUTO.GH.34")
        def test_01_action_taken_on_event(self):
            """Good description of test."""

            Aqf.step("Good description of first step")
            ...

eg. Using the decorator for a specific  system.::

    from nosekatreport import aqf_vr, system, Aqf, AqfBase

    @system('all')
    class TestAqf(AqfBase):

        @system('mkat', all=False)
        @aqf_vr("VR.CM.SITE.XX.12")
        def test_01_action_taken_on_event(self):
            """Good description of test."""

            Aqf.step("A MeerKAT specific test")
            ...


VR format
==========

Note the following with regards the VR format and its implication:

* The CAM system uses the VR format VR.CM.AUTO/DEMO/SITE.GGG.nnn
  where GGG is a grouping and nnn is a number.
* Tests are flagged with aqf_site_test, aqf_demo_test and aqf_auto_test from the VR name. 
  These test flags are passed to nose (by run_aqf.py) to select the tests to be included in the run
* QUALIFICATION TESTING event: includes all AUTO tests
* QUALIFICATION DEMONSTRATION event: includes all DEMO tests
* ACCEPTANCE TESTING event: includes all AUTO tests with @site_acceptance
* ACCEPTANCE DEMONSTRATION event: includes all DEMO tests with @site_acceptance and all SITE tests
* TO DO: What to do with @site_only tests, or is VR.CM.SITE.xxx enough to distinguish these tests.
* All these events include only the tests decorated for the current system (e.g. all, or mkat or kat7)



Available Aqf Decorators
========================

.. automodule:: nosekatreport

.. autofunction::  aqf_vr

.. autofunction::  system

.. autofunction::  slow

.. autofunction::  site_only

.. autofunction::  site_acceptance


Available Aqf Methods
=====================

.. #.. autoclass:: nosekatreport.Aqf
.. #        :members:


Example test
============

.. include:: nosekatreport/test_aqf.py
   :literal:

Ipython
========
AQF frame work can be used inside ipython shell. This will expose Aqf.sensors and allow you to interact and test sensors through the AQF like it would be done in a test.
In a nose test AQF inherits cam and sim objects from AqfBase class, when you call AQF in ipython you have to create cam and sim objects through katuilib.

Setup AQF in ipython after cam and sim objects has been created: ::

    from nosekatreport import Aqf
    Aqf.ipython()

Now Aqf sensor can be used in a similar manner as you would call it in a test: ::
    
    aqf_sensor = Aqf.sensor('sim.asc.sensor.wind_speed')
    val = aqf_sensor.get()
    aqf_sensor.set(val + 1)
    val = aqf_sensor.get()


Authors
=======

Neilen Marais

Lize van der Heever

Martin Slabber


Version History
===============

1.0
  * Updated for QBL(B) - separating out acceptance results and reports (katreport_accept) from qualification (katreport)
0.2
  * Stable and used in the Jan 2014 QBL.
0.1
  * Initial release
  * Hope it does something one day

