# Correlator-Beamforming Tests

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
* [_spead2_](https://github.com/ska-sa/spead2)  v1.1.1
 *   [_gcc4.9.3_](https://gcc.gnu.org/gcc-4.9/): spead2 dependency.

### Python testing dependencies packages

It is highly advisable to install these dependencies on a [_Python virtual environment_](https://virtualenv.pypa.io/), below is step-by-step instructions.
#### Setup and Use Virtualenv
```
# Install Python essentials and pip
$ sudo apt-get install python-dev build-essential
$ curl -s https://bootstrap.pypa.io/get-pip.py | python
$ sudo pip install virtualenv virtualenvwrapper

# Install testing dependencies in the virtualenv
$ bash scripts/setup_virtualenv.sh
```

### CBF Tests installation
Install CBF Tests (when virtualenv is active.).
```
$ python setup.py install
```

## Unit Testing

Running unit-testing.
```
$ . venv/bin/Activate
$ python run_cbf_tests.py --4A4k --no_slow
# For more options execute:
# python run_cbf_tests.py -h
```

The `python run_cbf_tests.py -h` script has several options. Please see `run_cbf_tests.py --help` for up-to-date detail.

```
(Test)mmphego@dbelab04:~/src/mkat_fpga_tests (AR2-test-improvements)
└─ [2017-09-08 14:57:47] $ >>> ./run_cbf_tests.py --help
Usage: 
        Usage: run_cbf_tests.py [options]
        This script auto executes CBF Tests with selected arguments.
        See Help for more information.
        

Options:
  -h, --help        show this help message and exit
  -v, --verbose     Be more verbose
  -q, --quiet       Be more quiet
  --nose=NOSE_ARGS  Additional arguments to pass on to nosetests.
                    eg: --nosetests "-x -s -v"
  --acceptance      Will only run test marked '@site_acceptance' or  if in the
                    Karoo(site) then also @site_only tests
  --dry_run         Do a dry run. Print commands that would be called
  --4A4k            Run the tests decorated with @instrument_bc8n856M4k
  --4A32k           Run the tests decorated with @instrument_bc8n856M32k
  --8A4k            Run the tests decorated with @instrument_bc16n856M4
  --8A32k           Run the tests decorated with @instrument_bc16n856M32k
  --16A4k           Run the tests decorated with @instrument_bc32n856M4k
  --16A32k          Run the tests decorated with @instrument_bc32n856M32k
  --quick           Only generate a small subset of the reports
  --no_html         Do not generate the html output
  --no_slow         Exclude tests decorated with @slow in this test run
  --report=REPORT   Only generate the reports. No tests will be run. Valid
                    options are: local, jenkins, skip and results. 'results'
                    will print the katreport[_accept].json test results
  --clean           Cleanup reports from previous test run. Reports
                    are replaced by default without --clean. Clean is
                    useful with --quick to only generate the html of the
                    test run report
  --dev_update      Do pip install update and install latest packages


```


## Contributors

 * Mpho Mphego
 * Alec Rust