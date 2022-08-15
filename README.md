# Correlator-Beamforming Tests

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/b8b5951e79a4414a85b450967f4faf2e)](https://app.codacy.com/app/mmphego/mkat_fpga_tests?utm_source=github.com&utm_medium=referral&utm_content=ska-sa/mkat_fpga_tests&utm_campaign=Badge_Grade_Dashboard)
[![Build Status](https://travis-ci.org/ska-sa/mkat_fpga_tests.svg?branch=devel)](https://travis-ci.org/ska-sa/mkat_fpga_tests)
[![LICENSE](https://img.shields.io/github/license/ska-sa/mkat_fpga_tests.svg?style=flat)](LICENSE)

A [Correlator-beamforming](https://www.ska.ac.za/science-engineering/meerkat/about-meerkat/) Unit and Acceptance Testing based framework for [MeerKAT](https://www.ska.ac.za/science-engineering/meerkat/) digital signal processing.

## Installation

Clone the repository including all submodules attached to it.

```shell
git clone --recursive git@github.com:ska-sa/mkat_fpga_tests.git
```

Also see: [opt/dsim_dependencies/README.md](opt/dsim_dependencies/README.md)

### Python Core packages

List of dependencies:

* [_katcp-python_](https://github.com/ska-sa/katcp-python)
* [_casperfpga_](https://github.com/ska-sa/casperfpga)
* [_corr2_](https://github.com/ska-sa/corr2)
* [_nosekatreport_](https://github.com/ska-sa/nosekatreport)
* [_spead2_](https://github.com/ska-sa/spead2)  v1.1.1
  *s [_gcc4.9.3_](https://gcc.gnu.org/gcc-4.9/): spead2 dependency.

### Python testing dependencies packages

It is highly recommended to install [_Python virtual environment_](https://virtualenv.pypa.io/) before continuing, else below is step-by-step instructions.

#### Setup and Use Virtualenv

```shell
# Install Python essentials and pip
curl -s https://bootstrap.pypa.io/get-pip.py | python
pip install --user -U virtualenv # or $ sudo pip install -U virtualenv

# Automagic installation of all dependencies in a virtualenv
make bootstrap
```

## Unit Testing

Running unit-testing.

```shell
# This will run all unit-tests defined in mkat_fpga_tests/test_cbf.py
make tests
```

## Acceptance Testing

The `python run_cbf_tests.py -h` script has several options. Please see `run_cbf_tests.py --help` for up-to-date detail.

```shell
(Test)mmphego@dbelab04:~/src/mkat_fpga_tests$ ./run_cbf_tests.py
usage: run_cbf_tests.py [-h] [--loglevel LOG_LEVEL] [-q] [--nose NOSE_ARGS]
                        [--acceptance SITE_ACCEPTANCE] [--instrument-activate]
                        [--dry_run] [--no-manual-test] [--available-tests]
                        [--4k] [--array_release_x] [--1k] [--32k] [--quick]
                        [--with_html] [--QTP] [--QTR] [--no_slow]
                        [--report REPORT] [--clean] [--dev_update]

This script auto executes CBF Tests with selected arguments.

optional arguments:
  -h, --help            show this help message and exit
  --loglevel LOG_LEVEL  log level to use, default INFO, options INFO, DEBUG,
                        WARNING, ERROR
  -q, --quiet           Be more quiet
  --nose NOSE_ARGS      Additional arguments to pass on to nosetests. eg:
                        --nosetests -x -s -v
  --acceptance SITE_ACCEPTANCE
                        Will only run test marked '@site_acceptance' or if in
                        the Karoo(site) then also @site_only tests
  --instrument-activate
                        launch an instrument. eg:./run_cbf_tests.py -v
                        --instrument-activate --4A4k
  --dry_run             Do a dry run. Print commands that would be called as
                        well as generatetest procedures
  --no-manual-test      Exclude manual tests decorated with @manual_test in
                        this test run
  --available-tests     Do a dry run. Print all tests available
  --4k                  Run the tests decorated with @instrument_4k
  --array_release_x     Run the tests decorated with @array_release_x
  --1k                  Run the tests decorated with @instrument_1k
  --32k                 Run the tests decorated with @instrument_32k
  --quick               Only generate a small subset of the reports
  --with_html           Generate HTML report output
  --QTP                 Generate PDF report output with Qualification Test
                        Procedure
  --QTR                 Generate PDF report output with Qualification Test
                        Report
  --no_slow             Exclude tests decorated with @slow in this test run
  --report REPORT       Only generate the reports. No tests will be run.Valid
                        options are: local, jenkins, skip and results.
                        'results' will print the katreport[_accept].json test
                        results
  --clean               Cleanup reports from previous test run. Reports are
                        replaced by default without --clean. Clean is useful
                        with --quick to only generate the html of the test run
                        report
  --dev_update          Do pip install update and install latest packages
  --revision REVISION   Specify QTR revision status number.
  --sensor_logs         Generates a log report of the sensor errors and
                        warnings occurred during the test run.
```

## Report Generation

For documentation we used [Sphinx](http://www.sphinx-doc.org/en/master/) and [latex](https://www.latex-project.org/), it is already included in the `pip-requirements.txt` to be installed.

*   See: [README.md](docs/Cover_Page/README.md)
*   See: [run_cbf_tests.py](https://github.com/ska-sa/mkat_fpga_tests/blob/devel/run_cbf_tests.py#L471)

The requirements for the qualification is pulled into the qualification test report from the MeerKAT.xml CORE file. Ensure that the lastest version of the CORE.xml file is copied to /usr/local/src/core_export on the host machine or to ./supplemental in the local repository.
 
The revision status number for each qualificaion test report is stored in .docs/rev_status.json and must be updated in order for any changes in the revision status to relfect in the QTR. The --revision argument may also be used to specify any other revision status number.

## TODO

* Report generation seems to be very tedious and needs some improvements.
    * see [run_cbf_tests.py](https://github.com/ska-sa/mkat_fpga_tests/blob/devel/run_cbf_tests.py#L471),
    * see [report.py](https://github.com/ska-sa/mkat_fpga_tests/blob/devel/report_generator/report.py#L16), this release doesn't need to be hard-coded.
    * see [process_core_xml.py](process_core_xml.py), this script converts `CORE.xml` (CORE export) into a json file to be used when extracting `REQ`, and etc which are then converted to `.rst` to be used by [report.py](report_generator/report.py) to generate a `latex` document... I am sure that can be improved or moved into it's own repository.
* Improve [test_cbf.py](mkat_fpga_tests/test_cbf.py), and [aqf_utils.py](mkat_fpga_tests/aqf_utils.py)

## Testing Philosophy

1. Test the common case of everything you can. This will tell you when that code breaks after you make some change (which is, in my opinion, the single greatest benefit of automated unit testing).
2. Test the edge cases of a few unusually complex code that you think will probably have errors.
3. Whenever you find a bug, write a test case to cover it before fixing it
4. Add edge-case tests to less critical code whenever someone has time to kill
5. “Code is like humour. When you have to explain it, it’s bad.” – Cory House

[Credit](https://softwareengineering.stackexchange.com/a/754)

## Contributors

* Mpho Mphego
* Alec Rust
