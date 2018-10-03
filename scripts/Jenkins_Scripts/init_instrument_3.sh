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
    gprint "Initialising Instrument ${RUN_INSTRUMENT} using kcs\n"
    . .venv/bin/activate
    which corr2_rx.py
    echo "backend: agg" > matplotlibrc
    [ -f "config/test_conf_site.ini" ] && sed -i -e 's/array0/array_0/g' config/test_conf_site.ini
    [ -f "scripts/instrument_activate" ] && bash scripts/instrument_activate ${RUN_INSTRUMENT} localhost array_0 y || exit 1;
else
    rprint "virtualenv not installed" && exit 1
fi