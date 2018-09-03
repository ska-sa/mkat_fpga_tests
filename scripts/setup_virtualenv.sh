#!/usr/bin/env bash

# Automated virtualenv and system packages installer which installs .venv in cwd
# Note: Include this file in the jenkins job script to setup the virtualenv.
# Author: Mpho Mphego <mmphego@ska.ac.za>

function usage() {
    echo -e "Usage: bash $0 VERBOSE SYS_PACKAGES\n
    Automated virtualenv and system packages installer which installs on cwd
    VERBOSE: Boolean:- if true, everything will be printed to stdout
    SYS_PACKAGES: Boolean:- if true, virtualenv will also use system packages\n";
    exit 1;
}

if [ -z "$*" ]; then
    usage
fi


# MAIN
VERBOSE=${1:-false}
SYS_PACKAGES=${2:-false}

if [ "${VERBOSE}" = true ]; then
    echo "Abort on any errors and verbose"
    set -ex
else
    set -e
fi

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

if [ "${SYS_PACKAGES}" = true ] ; then
    # Create virtual environment and include Python system packages
    gprint "Creating virtualenv venv directory and including system packages"
    if [ "${VERBOSE}" = true ]; then
        $(command -v virtualenv) "${VIRTUAL_ENV}" --system-site-packages
    else
        $(command -v virtualenv) "${VIRTUAL_ENV}" -q --system-site-packages
    fi
else
    gprint "Creating virtualenv venv directory without system packages"
    if [ "${VERBOSE}" = true ]; then
        $(command -v virtualenv) "${VIRTUAL_ENV}"
    else
        $(command -v virtualenv) "${VIRTUAL_ENV}" -q
    fi
fi

"$VIRTUAL_ENV"/bin/pip install -q -U pip setuptools
gprint "Sourcing virtualenv and exporting ${VIRTUAL_ENV}/bin to PATH..."
source "${VIRTUAL_ENV}/bin/activate"
export PATH="${VIRTUAL_ENV}/bin:$PATH"

gprint "Confirm that you are in a virtualenv: $(which python) \n\n"

if [ -z "${VIRTUAL_ENV}" ]; then
    rprint "Could not create virtualenv: $VIRTUAL_ENV"
    exit 2
fi

#
function install_pip_requirements() {
    FILENAME=$1                  # Filename to read requirements from.
    gprint "Installing development pip dependencies from ${FILENAME} file."
    if [ -f "$FILENAME" ]; then
        if [ "${VERBOSE}" = true ]; then
            $(command -v pip) install -r $FILENAME
        else
            $(command -v pip) install -q -r $FILENAME
        fi
    fi                           # do nothing if file is not found.
}


# You can ignore the https by using index-url and passing the http url as a parameter then set it as the trusted source.
# InsecurePlatformWarning: A true SSLContext object is not available.
# This prevents urllib3 from configuring SSL appropriately and may cause certain SSL connections to fail.
# You can upgrade to a newer version of Python to solve this.
if [ "${VERBOSE}" = true ]; then
    $(command -v pip) install --upgrade pip certifi pyOpenSSL ndg-httpsclient pyasn1 'requests[security]'
else
    $(command -v pip) install --quiet --upgrade pip certifi pyOpenSSL ndg-httpsclient pyasn1 'requests[security]'
fi

install_pip_requirements "pip-dev-requirements.txt"

if [ -f "./setup.py" ]; then
    gprint "Installing setup.py";
    # Install with dependencies.
    if [ "${VERBOSE}" = true ]; then
        python setup.py install -f;
    else
        python setup.py install -f > /dev/null 2>&1
    fi
fi
gprint "DONE!!!!\n\n"

if [ -f "scripts/pre_setup.sh" ]; then
    gprint "Install core dependencies, if pre_setup.sh script is available..."
    bash scripts/pre_setup.sh "${VERBOSE}"
fi