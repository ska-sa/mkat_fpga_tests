#!/bin/bash
set -xe #Verbose and abort on errors

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

    if ! "${RUN_TESTS}"; then
        gprint "Running Sanity Test based on Baseline Product Test\n"
        make sanitytest;
    else
        make tests4k;
    fi
else
    rprint "virtualenv not installed" && exit 1
fi


pylint --rcfile .pylintrc mkat_fpga_tests > katreport/pylint-report.txt || true;
