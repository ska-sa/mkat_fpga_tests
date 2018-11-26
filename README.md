# Correlator-Beamforming Tests

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/b8b5951e79a4414a85b450967f4faf2e)](https://app.codacy.com/app/mmphego/mkat_fpga_tests?utm_source=github.com&utm_medium=referral&utm_content=ska-sa/mkat_fpga_tests&utm_campaign=Badge_Grade_Dashboard)

A Correlator-beamforming unit-testing based framework for MeerKAT signal processing.

## Installation

### Debian packages

Install dependencies:

    $ sudo apt-get update && sudo apt-get install -yfm $(cat apt-build-requirements.txt)

### Python Core packages

Install dependencies to the system, by following their installation instructions:

* [_katcp-python_](https://github.com/ska-sa/katcp-python)
* [_casperfpga_](https://github.com/ska-sa/casperfpga)
* [_corr2_](https://github.com/ska-sa/corr2)
* [_nosekatreport_](https://github.com/ska-sa/nosekatreport)
* [_spead2_](https://github.com/ska-sa/spead2)  v1.1.1
 *   [_gcc4.9.3_](https://gcc.gnu.org/gcc-4.9/): spead2 dependency.

### Python testing dependencies packages

It is highly advisable to install these dependencies on a [_Python virtual environment_](https://virtualenv.pypa.io/), below is step-by-step instructions.
#### Setup and Use Virtualenv
```
# Install Python essentials and pip
$ curl -s https://bootstrap.pypa.io/get-pip.py | python
$ sudo pip install virtualenv virtualenvwrapper

# Install testing dependencies in the virtualenv
$ bash scripts/setup_virtualenv.sh

#Usage: bash scripts/setup_virtualenv.sh DEST_PATH SYS_PACKAGES
#
#    DEST_PATH: Workspace path
#    SYS_PACKAGES: Boolean:- if true, virtual env will also use system packages (Default:false)

```

### CBF Tests installation
Install CBF Tests 
```
$ . venv/bin/activate
$ python setup.py install
```

## Unit Testing

Running unit-testing.
```
$ . venv/bin/activate
$ python run_cbf_tests.py --4A4k --no_slow
# For more options execute:
# python run_cbf_tests.py -h
```

The `python run_cbf_tests.py -h` script has several options. Please see `run_cbf_tests.py --help` for up-to-date detail.

```
(Test)mmphego@dbelab04:~/src/mkat_fpga_tests$ ./run_cbf_tests.py 
Usage: 
        Usage: run_cbf_tests.py [options]
        This script auto executes CBF Tests with selected arguments.
        See Help for more information.

Options:
  -h, --help            show this help message and exit
  --loglevel=LOG_LEVEL  log level to use, default INFO, options INFO, DEBUG,
                        WARNING, ERROR
  -q, --quiet           Be more quiet
  --nose=NOSE_ARGS      Additional arguments to pass on to nosetests.
                        eg: --nosetests "-x -s -v"
  --acceptance          Will only run test marked '@site_acceptance' or  if in
                        the Karoo(site) then also @site_only tests
  --instrument-activate
                        launch an instrument. eg:
                        ./run_cbf_tests.py -v --instrument-activate --4A4k
  --dry_run             Do a dry run. Print commands that would be called as
                        well as generate test procedures
  --available-tests     Do a dry run. Print all tests available
  --4A4k                Run the tests decorated with @instrument_bc8n856M4k
  --4A32k               Run the tests decorated with @instrument_bc8n856M32k
  --8A4k                Run the tests decorated with @instrument_bc16n856M4
  --8A32k               Run the tests decorated with @instrument_bc16n856M32k
  --16A4k               Run the tests decorated with @instrument_bc32n856M4k
  --16A32k              Run the tests decorated with @instrument_bc32n856M32k
  --quick               Only generate a small subset of the reports
  --with_html           Generate HTML report output
  --QTP                 Generate PDF report output with Qualification Test
                        Procedure
  --QTR                 Generate PDF report output with Qualification Test
                        Results
  --no_slow             Exclude tests decorated with @slow in this test run
  --report=REPORT       Only generate the reports. No tests will be run. Valid
                        options are: local, jenkins, skip and results.
                        'results' will print the katreport[_accept].json test
                        results
  --clean               Cleanup reports from previous test run. Reports
                        are replaced by default without --clean. Clean is
                        useful with --quick to only generate the html of the
                        test run report
  --dev_update          Do pip install update and install latest packages
```

## Testing Philosophy 

1. Test the common case of everything you can. This will tell you when that code breaks after you make some change (which is, in my opinion, the single greatest benefit of automated unit testing).
2. Test the edge cases of a few unusually complex code that you think will probably have errors.
3. Whenever you find a bug, write a test case to cover it before fixing it
4. Add edge-case tests to less critical code whenever someone has time to kil
Credit: https://softwareengineering.stackexchange.com/a/754
5. “Code is like humour. When you have to explain it, it’s bad.” – Cory House


## Contributors

 * Mpho Mphego
 * Alec Rust