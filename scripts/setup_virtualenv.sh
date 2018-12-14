#!/usr/bin/env bash

# Automated virtualenv and system packages installer which installs .venv in cwd
# Note: Include this file in the jenkins job script to setup the virtualenv.
# Author: Mpho Mphego <mmphego@ska.ac.za>

set -e

RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
NORMAL=$(tput sgr0)

function gprint (){
    printf "%s$1%s\n" "${GREEN}" "${NORMAL}";
}

function rprint (){
    printf "%s$1%s\n" "${RED}" "${NORMAL}";
}

# ---------------------
VIRTUAL_ENV=".venv"

gprint "Installing ${VIRTUAL_ENV} in current working directory"
$(command -v virtualenv) "${VIRTUAL_ENV}" -q

gprint "Sourcing virtualenv and exporting ${VIRTUAL_ENV}/bin to PATH..."
PYVENV="${VIRTUAL_ENV}/bin/python"
"${PYVENV}" -W ignore::Warning -m pip install -q -U pip setuptools wheel
export PATH="${VIRTUAL_ENV}/bin:$PATH"

gprint "Confirm that you are in a virtualenv: $(which python)"

if [ -z "${VIRTUAL_ENV}" ]; then
    rprint "Could not create virtualenv: ${VIRTUAL_ENV}"
    exit 2
fi

#
function install_pip_requirements() {
    FILENAME=$1                  # Filename to read requirements from.
    gprint "Installing development pip dependencies from ${FILENAME} file."
    if [ -f "${FILENAME}" ]; then
        "${PYVENV}" -W ignore::Warning -m pip install -q -r "${FILENAME}"
    fi
}

"${PYVENV}" -W ignore::Warning -m pip install --quiet --upgrade \
    pip numpy certifi pyOpenSSL ndg-httpsclient pyasn1 'requests[security]'

env CC=$(which gcc) CXX=$(which g++) "${PYVENV}" -W ignore::Warning -m pip wheel --no-cache-dir \
     https://github.com/ska-sa/spead2/releases/download/v1.2.0/spead2-1.2.0.tar.gz

function post_setup(){
    if [ -f "setup.py" ]; then
        gprint "Installing setup.py";
        # Install with dependencies.
        "${PYVENV}" setup.py install -f;
    fi
}

function check_packages(){
    declare -a PYTHON_PACKAGES=(spead2 casperfpga corr2 nosekatreport)
    for pkg in "${PYTHON_PACKAGES[@]}";do
        gprint "Checking ${pkg} if it is installed!";
        "${PYVENV}" -c "import ${pkg}";
        if [ "$?" = 0 ]; then
            rprint "${pkg} is not installed"
            exit 1
        else
            exit 0
        fi

    done
}
post_setup
install_pip_requirements "pip-dev-requirements.txt"
check_packages
rm -rf -- *.whl
