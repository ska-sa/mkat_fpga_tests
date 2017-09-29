#!/bin/bash
# Automated system packages installer
# Note: Include this file in the jenkins job script to setup the virtualenv.
# Author: Mpho Mphego <mmphego@ska.ac.za>


# Abort on any errors
set -e

if [ -z "$*" ];
    then echo -e "Usage: bash $0 DEST_PATH SYS_PACKAGES\n
    DEST_PATH: Workspace path
    SYS_PACKAGES: Boolean:- if true, virtual env will also use system packages (Default:false)\n";
    exit 1;
fi

# MAIN
DEST_PATH=$1
SYS_PACKAGES=${2:-false}

# ---------------------
echo -e "Working in: ${DEST_PATH}\n"
cd "${DEST_PATH}"
# Create virtual enviroment and include Python system packages
if [ "${SYS_PACKAGES}" = true ] ; then
    echo -e "Creating virtualenv venv directory and including system packages\n"
    virtualenv venv --system-site-packages
else
    echo -e "Creating virtualenv venv directory without system packages\n"
    virtualenv venv
fi
source ./venv/bin/activate

if [ -z "${VIRTUAL_ENV}" ]; then
    echo -e "Could not create virtual env. $VIRTUAL_ENV\n"
    exit 2
fi

#
function install_pip_requirements {
    FILENAME=$1                  # Filename to read requirements from.
    WARNING=$2                   # If this is a depricated filename give a warning.
    if [ -f "$FILENAME" ]; then
        if $WARNING; then
            echo "!!! The use of $FILENAME is depricated."
            echo "!!! Please put you PIP build requirements"
            echo "!!! into pip-requirements.txt"
        fi
        # pip install --trusted-host pypi.camlab.kat.ac.za --pre -r $FILENAME
        cat $FILENAME | grep -v -e "^$" -e "^#" | sort -u | while read line
        do
            python "$(which pip)" install --no-cache-dir --quiet --index-url=http://pypi.python.org/simple/ --trusted-host pypi.python.org $line
            # python "$(which pip)" install --no-cache-dir -q --trusted-host pypi.camlab.kat.ac.za $line
            # python "$(which pip)" install --trusted-host pypi.camlab.kat.ac.za --pre $line
            echo -n "."
        done
    fi                           # do nothing if file is not found.
}

# You can ignore the https by using index-url and passing the http url as a parameter then set it as the trusted source.
# InsecurePlatformWarning: A true SSLContext object is not available.
# This prevents urllib3 from configuring SSL appropriately and may cause certain SSL connections to fail.
# You can upgrade to a newer version of Python to solve this.
python "$(which pip)" install --no-cache-dir --quiet --upgrade pip certifi pyOpenSSL ndg-httpsclient pyasn1 'requests[security]'
# Install pip dependencies
install_pip_requirements pip-requirements.txt true
# Install core dependencies if pre_setup.sh script is available
PRE_SETUP="${DEST_PATH}/pre_setup.sh"
if [ -f "${PRE_SETUP}" ]
then
    "${PRE_SETUP}"
fi


# DEPRECATED, setup.py does this automagically
# Install Self.
# If the given DEST_PATH contains a setup.py we will install it.
# Previously had the install of self in the pip-build-requirements.txt
# as a line with a '.'. That worked but on occasion we got wierd errors.
# SETUP="${DEST_PATH}/setup.py"
# if [ -f "${SETUP}" ]; then
#     cd "${DEST_PATH}"
#     echo -e "Installing setup.py\n"
#     # Install with dependencies.
#     python "$(which pip)" install --trusted-host pypi.camlab.kat.ac.za --pre .
# fi
