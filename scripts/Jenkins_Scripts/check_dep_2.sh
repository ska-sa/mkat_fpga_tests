#!/bin/bash
set -xe #Verbose and abort on errors
export PATH=$PATH:$WORKSPACE/scripts:$WORKSPACE/.venv/bin

RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
NORMAL=$(tput sgr0)

rprint() {
    STRING=$1
    printf "%s${STRING}%s\n" "${RED}" "${NORMAL}"
}
gprint() {
    STRING=$1
    printf "%s${STRING}%s\n" "${GREEN}" "${NORMAL}"
}

if [ -f ".venv/bin/activate" ]; then
    . .venv/bin/activate > /dev/null 2>&1
    gprint "Sanity Check...";
    which corr2_dsim_control.py;
    declare -a PYTHON_SRC_PACKAGES="(casperfpga corr2 spead2 katcp)"
    for pkg in "${PYTHON_SRC_PACKAGES[@]}"; do
        $(command -v python) -c """import "$pkg"; print $pkg.__file__;"""
        [ $? -eq 0 ] && gprint "${pkg} installed successfully in virtualenv!" || exit $?;
    done
else
    rprint "virtualenv not installed" && exit 1
fi