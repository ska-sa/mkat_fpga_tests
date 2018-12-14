#!/usr/bin/env bash

# Automated virtualenv and system packages installer which installs .venv in cwd
# Note: Include this file in the jenkins job script to setup the virtualenv.
# Author: Mpho Mphego <mmphego@ska.ac.za>

# function usage() {
#     echo -e "Usage: bash $0 VERBOSE SYS_PACKAGES\n
#     Automated virtualenv and system packages installer which installs on cwd
#     VERBOSE: Boolean:- if true, everything will be printed to stdout
#     SYS_PACKAGES: Boolean:- if true, virtualenv will also use system packages\n";
#     exit 1;
# }

# if [ -z "$*" ]; then
#     usage
# fi


# # MAIN
# VERBOSE=${1:-false}
# SYS_PACKAGES=${2:-false}

# if [ "${VERBOSE}" = true ]; then
#     echo "Abort on any errors and verbose"
#     set -ex
# else
#     set -e
# fi
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
source "${VIRTUAL_ENV}/bin/activate"
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
    pip certifi pyOpenSSL ndg-httpsclient pyasn1 'requests[security]'

"${PYVENV}" -W ignore::Warning -m pip install -U numpy
env CC=$(which gcc) CXX=$(which g++) "${PYVENV}" -W ignore::Warning -m pip wheel --no-cache-dir \
     https://github.com/ska-sa/spead2/releases/download/v1.2.0/spead2-1.2.0.tar.gz

function post_setup(){
    if [ -f "setup.py" ]; then
        gprint "Installing setup.py";
        # Install with dependencies.
        "${PYVENV}" setup.py install -f;
    fi
}

# pre_setup
post_setup
install_pip_requirements "pip-dev-requirements.txt"
#gprint "DONE!!!!\n\n"
#bash --rcfile "${VENV}/bin/activate" -i
