#!/bin/bash
set -e #Verbose and abort on errors
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

gprint "Resetting templaces in /etc/corr"
cd /etc/corr/templates || true; 
git diff --exit-code "${RUN_INSTRUMENT}" || git checkout -- "${RUN_INSTRUMENT}"; 
cd - || true;
gprint "Deleting old fpgs stored in tmp"
rm -rf /tmp/*.fpg || true;

gprint "Installing virtualenv and all dependencies"
if [ -f "scripts/setup_virtualenv.sh" ]; then
    bash scripts/setup_virtualenv.sh true
    export PATH=$PATH:$WORKSPACE/scripts:$WORKSPACE/.venv/bin;

    . .venv/bin/activate > /dev/null 2>&1;      
else
    rprint "Failed to create virtualenv and install dependencies"
    exit 1;
fi

