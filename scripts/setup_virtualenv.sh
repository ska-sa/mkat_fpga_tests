#!/usr/bin/env bash

# Automated system packages installer
# Note: Include this file in the jenkins job script to setup the virtualenv.
# Author: Mpho Mphego <mmphego@ska.ac.za>


# Abort on any errors
set -e

RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
NORMAL=$(tput sgr0)

if [ -z "$*" ];
    then echo -e "Usage: bash $0 SYS_PACKAGES\n
    Automated virtualenv and system packages installer which installs on cwd
    SYS_PACKAGES: Boolean:- if true, virtualenv will also use system packages (Default:false)\n";
    exit 1;
fi

# MAIN
SYS_PACKAGES=${1:-false}

# ---------------------
VIRTUAL_ENV=".venv"
printf "${GREEN}Installing ${VIRTUAL_ENV} in current working directory${NORMAL}\n"
if [ "${SYS_PACKAGES}" = true ] ; then
    # Create virtual environment and include Python system packages
    printf "${GREEN}Creating virtualenv venv directory and including system packages${NORMAL}\n"
    $(which virtualenv) "$VIRTUAL_ENV" -q --system-site-packages
else
    printf "${GREEN}Creating virtualenv venv directory without system packages${NORMAL}\n"
    $(which virtualenv) "$VIRTUAL_ENV" -q
fi
"$VIRTUAL_ENV"/bin/pip install -q -U pip setuptools
printf "${GREEN}Sourcing virtualenv and exporting ${VIRTUAL_ENV}/bin to PATH...${NORMAL}\n"
source "${VIRTUAL_ENV}/bin/activate"
export PATH="${VIRTUAL_ENV}/bin:$PATH"

printf "${GREEN}Confirm that you are in a virtualenv: $(which python) ${NORMAL}\n\n\n"

if [ -z "${VIRTUAL_ENV}" ]; then
    printf "${RED}Could not create virtualenv: $VIRTUAL_ENV${NORMAL}\n"
    exit 2
fi

#
function install_pip_requirements() {
    FILENAME=$1                  # Filename to read requirements from.
    if [ -f "$FILENAME" ]; then
        if $SYS_PACKAGES; then
            printf "${RED}!!! The use of $FILENAME is depricated.${NORMAL}\n"
            printf "${RED}!!! Please put you PIP build requirements${NORMAL}\n"
            printf "${RED}!!! into pip-dev-requirements.txt${NORMAL}\n\n"
        fi
        pip install -r $FILENAME
    fi                           # do nothing if file is not found.
}

if "${SYS_PACKAGES}"; then

    # You can ignore the https by using index-url and passing the http url as a parameter then set it as the trusted source.
    # InsecurePlatformWarning: A true SSLContext object is not available.
    # This prevents urllib3 from configuring SSL appropriately and may cause certain SSL connections to fail.
    # You can upgrade to a newer version of Python to solve this.
    pip install --quiet --upgrade pip certifi pyOpenSSL ndg-httpsclient pyasn1 'requests[security]'
    printf "${GREEN}Installing development pip dependencies, with system site packages${NORMAL}\n"
    install_pip_requirements pip-dev-requirements.txt
    printf "${GREEN}DONE...${NORMAL}\n"

    if [ -f "./setup.py" ]; then
        printf "${GREEN}Installing setup.py${NORMAL}\n";
        # Install with dependencies.
        python setup.py install -f;
    fi
fi

# if [ -f "scripts/pre_setup.sh" ]; then
#     printf "${GREEN}Install core dependencies, if pre_setup.sh script is available${NORMAL}\n\n"
#     bash scripts/pre_setup.sh
# fi