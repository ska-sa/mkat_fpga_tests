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
if ! $(command -v virtualenv) >/dev/null; then
    pip install --user virtualenv virtualenvwrapper
fi
$(command -v virtualenv) "${VIRTUAL_ENV}" --download --system-site-packages

gprint "Sourcing virtualenv and exporting ${VIRTUAL_ENV}/bin to PATH..."
source "${VIRTUAL_ENV}/bin/activate"
if [ -d "/opt/gcc4.9.3/bin" ]; then
    export PATH=/opt/gcc4.9.3/bin:"${VIRTUAL_ENV}/bin:$PATH"
    export LD_LIBRARY_PATH=/opt/gcc4.9.3/lib64/:/opt/gcc4.9.3/lib32/:/usr/lib/x86_64-linux-gnu/:"${LD_LIBRARY_PATH}"
else
    export PATH="${VIRTUAL_ENV}/bin:$PATH"
    export LD_LIBRARY_PATH=/usr/lib/gcc/x86_64-linux-gnu/:/usr/lib/x86_64-linux-gnu/:"${LD_LIBRARY_PATH}"
fi
$(command -v pip) install -U pip setuptools wheel
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

function install_pip_dependencies() {
    pip install --upgrade \
        certifi pyOpenSSL ndg-httpsclient pyasn1 'requests[security]' numpy>1.15.0 tornado==4.*

    # Last tested working spead2.
    # env CC="ccache gcc" CXX="g++" $(command -v python) -W ignore::Warning -m pip wheel --no-cache-dir \
    #     https://github.com/ska-sa/spead2/releases/download/v1.2.0/spead2-1.2.0.tar.gz
    # if [ -f "spead2-1.2.0-cp27-cp27mu-linux_x86_64.whl" ]; then
    #     env CC="ccache gcc" CXX="g++" $(command -v python) -W ignore::Warning -m pip install \
    #         spead2-1.2.0-cp27-cp27mu-linux_x86_64.whl
    # fi
    # pkg_checker spead2
    # cd opt/spead2
    # env CC="ccache gcc" CXX="g++" PATH=$PATH $(command -v pip) install .
    # cd -

    # Installing nosekatreport
    pip install -I \
        git+https://github.com/ska-sa/nosekatreport.git@karoocbf#egg=nosekatreport

    # Installing casperfpga
    pip install -v --no-dependencies -I \
        git+https://github.com/ska-sa/casperfpga@devel#egg=casperfpga

    # Installing corr2 and manually installing dependencies
    pip install -v --no-dependencies -I \
        git+https://github.com/ska-sa/corr2@devel#egg=corr2
}

function post_setup(){
    [ -f "setup.py" ] && $(command -v python) setup.py install -f
}

function verify_pkgs_installed(){
    declare -a pkgs=("corr2" "casperfpga" "katcp" "nosekatreport")
    for pkg in "${pkgs[@]}"; do
        pkg_checker "${pkg}";
    done
}

install_pip_requirements "pip-dev-requirements.txt"
install_pip_dependencies
post_setup
verify_pkgs_installed