#!/bin/bash

set -e # Abort on any errors

. venv/bin/activate

# Special spead2 hacksauce for dbelab04 with old GCC. This will be
# fixed properly by upgrading the test system GCC
#
DIRECTORY="$PYTHON_SRC_PACKAGES_DIR"/spead2

#cd "$PYTHON_SRC_PACKAGES_DIR"/spead2
#cd $DIRECTORY
if [ -d "$DIRECTORY" ]; then
    cd $DIRECTORY
else
    cd "$PYTHON_SRC_PACKAGES_DIR";
    wget https://pypi.python.org/packages/a1/0f/9cf4ab8923a14ff349d5e85c89ec218ab7a790adfdcbd11877393d0c5bba/spead2-1.1.1.tar.gz
    tar -zxvf spead2-1.1.1.tar.gz
    mv spead2-1.1.1 spead2
    cd $DIRECTORY
fi

OLD_PATH="$PATH"
export PATH=/home/paulp/opt/gcc4.9.3/bin:"${PATH}"
pip install -e . --force-reinstall
export PATH="${OLD_PATH}"
# end hacksauce
##
set -u # Throw error on uninitialized variables
