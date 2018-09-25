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
    kcpcmd -t 60 -s localhost:$(kcpcmd array-list array_0 | grep -a array-list | cut -f3 -d ' ' | cut -f1 -d',') capture-start baseline-correlation-products || true;
    timeout 60s .venv/bin/python .venv/bin/corr2_rx.py --loglevel INFO --config "/etc/corr/array_0-${RUN_INSTRUMENT}" --print  --warmup_cap --baseline 4  --channels 0,2047 || true;
else
    rprint "virtualenv not installed" && exit 1
fi

