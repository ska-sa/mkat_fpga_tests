# Correlator-Beamforming Tests

A Correlator-beamforming unit-testing based framework for MeerKAT signal processing.

## Installation

### Debian packages

Install dependencies:

    $ sudo bash apt-build-dependencies.sh

### Python Core packages

Install dependencies to the system, by following their installation instructions:

* [_katcp-python_](https://github.com/ska-sa/katcp-python)
* [_casperfpga_](https://github.com/ska-sa/casperfpga)
* [_corr2_](https://github.com/ska-sa/corr2)
* [_spead2_](https://github.com/ska-sa/spead2)  v1.1.1 
 *   [_gcc4.9.3_](https://gcc.gnu.org/gcc-4.9/): spead2 dependency.

### Python testing dependencies packages

It is highly advisable to install these dependencies on a [_Python virtual environment_](https://virtualenv.pypa.io/), below is step-by-step instructions.
####Setup and Use Virtualenv
```
# Install Python essentials and pip
$ sudo apt-get install python-dev build-essential
$ curl -s https://bootstrap.pypa.io/get-pip.py | python
$ sudo pip install virtualenv virtualenvwrapper

# Activate Virtualenv
$ mkdir -p ~/virtualenvironment && cd "$_"
$ virtualenv venv
$ . ./venv/bin/activate

# Install testing dependencies in the virtualenv
$ bash scripts/setup_virtualenv.sh
$ pip install --no-cache-dir -r pip-requirements.txt
# --no-cache-dir: Disable the cache.
# -r : Install from the given requirements file.
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

## Contributors

 * Mpho Mphego
 * Alec Rust