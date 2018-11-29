# Correlator-Beamforming Tests

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/b8b5951e79a4414a85b450967f4faf2e)](https://app.codacy.com/app/mmphego/mkat_fpga_tests?utm_source=github.com&utm_medium=referral&utm_content=ska-sa/mkat_fpga_tests&utm_campaign=Badge_Grade_Dashboard)

A [Correlator-beamforming](https://www.ska.ac.za/science-engineering/meerkat/about-meerkat/) Unit and Acceptance Testing based framework for [MeerKAT](https://www.ska.ac.za/science-engineering/meerkat/) digital signal processing.

## Installation

Also see: `opt/dsim_dependencies/README.md`

### Python Core packages

List of dependencies:

* [_katcp-python_](https://github.com/ska-sa/katcp-python)
* [_casperfpga_](https://github.com/ska-sa/casperfpga)
* [_corr2_](https://github.com/ska-sa/corr2)
* [_nosekatreport_](https://github.com/ska-sa/nosekatreport)
* [_spead2_](https://github.com/ska-sa/spead2)  v1.1.1
 *   [_gcc4.9.3_](https://gcc.gnu.org/gcc-4.9/): spead2 dependency.

### Python testing dependencies packages

It is highly recommended to install [_Python virtual environment_](https://virtualenv.pypa.io/) before continuing, else below is step-by-step instructions.

#### Setup and Use Virtualenv
```
# Install Python essentials and pip
$ curl -s https://bootstrap.pypa.io/get-pip.py | python
$ pip install --user -U virtualenv # or $ sudo pip install -U virtualenv

# Automagic installation of all dependencies in a virtualenv
$ make bootstrap
```

## Unit Testing

Running unit-testing.
```
# This will run all unit-tests defined in mkat_fpga_tests/test_cbf.py
$ make tests
```

## Acceptance Testing

The `python run_cbf_tests.py -h` script has several options. Please see `run_cbf_tests.py --help` for up-to-date detail.

```
(Test)mmphego@dbelab04:~/src/mkat_fpga_tests$ ./run_cbf_tests.py
usage: run_cbf_tests.py [-h] [--loglevel LOG_LEVEL] [-q] [--nose NOSE_ARGS]
                        [--acceptance SITE_ACCEPTANCE] [--instrument-activate]
                        [--dry_run] [--no-manual-test] [--available-tests]
                        [--4k] [--32k] [--quick] [--with_html] [--QTP] [--QTR]
                        [--no_slow] [--report REPORT] [--clean] [--dev_update]

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
```

## Testing Philosophy

1. Test the common case of everything you can. This will tell you when that code breaks after you make some change (which is, in my opinion, the single greatest benefit of automated unit testing).
2. Test the edge cases of a few unusually complex code that you think will probably have errors.
3. Whenever you find a bug, write a test case to cover it before fixing it
4. Add edge-case tests to less critical code whenever someone has time to kill
Credit: https://softwareengineering.stackexchange.com/a/754
5. “Code is like humour. When you have to explain it, it’s bad.” – Cory House

## Contributors

 * Mpho Mphego
 * Alec Rust
