#!/usr/bin/env bash

# Automated virtualenv and system packages installer which installs .venv in cwd
# Note: Include this file in the jenkins job script to setup the virtualenv.
# Author: Mpho Mphego <mmphego@ska.ac.za>

set -e pipefail

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


"${VIRTUAL_ENV}"/bin/python -W ignore::Warning -m pip install -q -U pip setuptools wheel
gprint "Sourcing virtualenv and exporting ${VIRTUAL_ENV}/bin to PATH..."
source "${VIRTUAL_ENV}/bin/activate"
if [ -d "/opt/gcc4.9.3/bin" ]; then
    export PATH=/opt/gcc4.9.3/bin:"${VIRTUAL_ENV}/bin:$PATH"
else
    export PATH="${VIRTUAL_ENV}/bin:$PATH"
fi
gprint "Confirm that you are in a virtualenv: $(which python)"

if [ -z "${VIRTUAL_ENV}" ]; then
    rprint "Could not create virtualenv: ${VIRTUAL_ENV}"
    exit 2
fi

#
function pkg_checker() {
    pkg=$1
    if ! $(command -v python) -c "import ${pkg}; print ${pkg}.__file__" >/dev/null; then
        echo "Failed to install ${pkg}";
        exit 1;
    fi
}

function install_pip_requirements() {
    FILENAME=$1                  # Filename to read requirements from.
    gprint "Installing development pip dependencies from ${FILENAME} file."
    if [ -f "$FILENAME" ]; then
        $(command -v python) -W ignore::Warning -m pip install --no-cache-dir --ignore-installed -r "${FILENAME}" || true
    fi
}

function pip_dependencies() {
    $(command -v python) -W ignore::Warning -m pip install --quiet --upgrade pip certifi pyOpenSSL \
        ndg-httpsclient pyasn1 'requests[security]'

    $(command -v python) -W ignore::Warning -m pip install --quiet --ignore-installed --no-cache-dir \
        numpy>=1.15.0
    # Last tested working spead2.
    env CC=$(which gcc) CXX=$(which g++) $(command -v python) -W ignore::Warning -m pip wheel --no-cache-dir \
        https://github.com/ska-sa/spead2/releases/download/v1.2.0/spead2-1.2.0.tar.gz
    if [ -f "spead2-1.2.0-cp27-cp27mu-linux_x86_64.whl" ]; then
        env CC=$(which gcc) CXX=$(which g++) $(command -v python) -W ignore::Warning -m pip install \
            spead2-1.2.0-cp27-cp27mu-linux_x86_64.whl
    fi

    # Installing katcp-python
    $(command -v python) -W ignore::Warning -m pip install --force-reinstall \
        tornado>=4.3 katcp

    # Installing nosekatreport
    $(command -v python) -W ignore::Warning -m pip install \
        git+https://github.com/ska-sa/nosekatreport.git@karoocbf#egg=nosekatreport

    # Installing casperfpga
    $(command -v python) -W ignore::Warning -m pip install -I --no-deps \
        odict \
        git+https://github.com/ska-sa/casperfpga@devel#egg=casperfpga

    # Installing corr2 and manually installing dependencies
    $(command -v python) -W ignore::Warning -m pip install -I --no-deps \
        h5py \
        iniparse \
        matplotlib==2.0.2 \
        coloredlogs \
        lazy-import>=0.2 \
        git+https://github.com/ska-sa/corr2@devel#egg=corr2
}

function post_setup(){
if [ -f "setup.py" ]; then
    gprint "Installing setup.py";
    # Install with dependencies.
    $(command -v python) setup.py install -f;
fi
}

function verify_pkgs_installed(){
    declare -a pkgs=("corr2" "casperfpga" "katcp" "nosekatreport" "spead2")
    for pkg in "${pkgs[@]}"; do
        pkg_checker "${pkg}";
    done
}

pip_dependencies
install_pip_requirements "pip-dev-requirements.txt"
post_setup
verify_pkgs_installed