#!/usr/bin/env bash

# Automated python dependencies retrieval and installation,
# This script will install all CBF-Testing dependencies namely:-
#       spead2, corr2, casperfpga and katcp-python
# These packages will be cloned from their git repositories and installed in /usr/local/src which
# is a dependency default directory for CBF-Testing, and assuming that the latest CMC Software is installed and functional

# Author: Mpho Mphego <mmphego@ska.ac.za>

RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
NORMAL=$(tput sgr0)
CURDIR="${PWD}"
VENV=".venv/bin/activate"

VERBOSE=${1:-false}
if [ "${VERBOSE}" = true ]; then
    set -ex
else
    set -e
fi

function gprint (){
    printf "%s$1%s\n" "${GREEN}" "${NORMAL}";
}

function rprint (){
    printf "%s$1%s\n" "${RED}" "${NORMAL}";
}

if [ -f "${VENV}" ]; then

    source "${VENV}"
    # Special spead2 hacksauce for dbelab04 with old GCC. This will be
    # fixed properly by upgrading the test system GCC
    export PATH=/opt/gcc4.9.3/bin:"${PWD}/.venv/bin":"${PATH}"

    declare -a PYTHON_SRC_PACKAGES=(spead2 casperfpga corr2 katcp-python)
    PYTHON_SRC_DIR=/usr/local/src
    SPEAD2_URL=https://pypi.python.org/packages/a1/0f/9cf4ab8923a14ff349d5e85c89ec218ab7a790adfdcbd11877393d0c5bba/spead2-1.1.1.tar.gz
    PYTHON_SETUP_FILE=setup.py

    function pip_installer {
        pkg="$1"
        export PYTHON_PKG="${pkg}"

        if [ "${pkg}" = 'katcp-python' ]; then
            pkg="katcp"
        fi

        if $(command -v python) -c "import os; pypkg = os.environ['PYTHON_PKG']; __import__(pypkg)" &> /dev/null; then
            gprint "${pkg} Package already installed";
        else
            gprint "Installing ${pkg} in ${INSTALL_DIR}"
            cd "${INSTALL_DIR}"
            if [ ! -f "${PYTHON_SETUP_FILE}" ]; then
                rprint "Python ${PYTHON_SETUP_FILE} file not found!\n"
                continue
            else
                if [ "${VERBOSE}" = true ]; then
                    $(command -v pip) install -e .
                else
                    $(command -v pip) install -q -e .
                fi
                # NO SUDOing when automating
                # sudo python setup.py install --force
                gprint "Successfully installed ${pkg} in ${INSTALL_DIR}"
            fi
        fi
    }

    function spead2_installer {
        pkg="$1"
        cd "${INSTALL_DIR}"
        gprint "Installing ${pkg}"
        # NO SUDOing when automating
        # env PATH=$PATH sudo pip install -v .
        if [ "${VERBOSE}" = true ]; then
            $(command -v pip) install -v .
        else
            $(command -v pip) install -q .
        fi
        gprint "Successfully installed ${pkg} in ${INSTALL_DIR}"
    }

    for pkg in "${PYTHON_SRC_PACKAGES[@]}"; do
        INSTALL_DIR="${PYTHON_SRC_DIR}"/"${pkg}"
        printf "\n\n"
        if python -c "import os; pypkg = os.environ['PYTHON_PKG']; __import__(pypkg)" &> /dev/null; then
            rprint "${pkg} Package already installed";
        elif [ "${pkg}" = 'spead2' ]; then
            if [ -d "${INSTALL_DIR}" ]; then
                gprint "${pkg} directory exists."
                export PYTHON_PKG="${pkg}"
                if python -c "import os; pypkg = os.environ['PYTHON_PKG']; __import__(pypkg)" &> /dev/null; then
                    rprint '${pkg} Package already installed';
                else
                    spead2_installer "${pkg}"
                fi
            else
                rprint "${pkg} directory doesnt exist cloning."
                mkdir -p "${INSTALL_DIR}" && cd "$_"
                curl -s "${SPEAD2_URL}" | tar zx
                mv "${pkg}"* "${pkg}"
                export PYTHON_PKG="${pkg}"
                if python -c "import os; pypkg = os.environ['PYTHON_PKG']; __import__(pypkg)" &> /dev/null; then
                    gprint "Package ${pkg} already installed";
                else
                    spead2_installer "${pkg}"
                fi
            fi
        elif [ -d "${INSTALL_DIR}" ]; then
            gprint "${pkg} directory exists."
            pip_installer "${pkg}"
        else
            rprint "${pkg} directory doesnt exist cloning."
            $(command -v git) clone git@github.com:ska-sa/"${pkg}".git "${PYTHON_SRC_DIR}"/"${pkg}" && cd "$_"
            pip_installer "${pkg}"
        fi
        $(command -v python) -c "import $pkg; print $pkg.__file__" || true
    done
    cd "${CURDIR}"
    bash --rcfile "${VENV}" -i
else
    gprint "VIRTUAL ENVIROMENT MISSING!!!!!!!!!!!!!\n"
fi