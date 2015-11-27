#!/bin/bash

set -e # Abort on any errors

PYTHON_SRC_PACKAGES=(katcp-python casperfpga corr2)

. venv/bin/activate

# Special spead2 hacksauce for dbelab04 with old GCC. This will be
# fixed properly by upgrading the test system GCC
#
cd "$PYTHON_SRC_PACKAGES_DIR"/spead2
OLD_PATH="$PATH"
export PATH=/home/paulp/opt/gcc4.9.3/bin:"${PATH}"
#pip install -e .
export PATH="${OLD_PATH}"
# end hacksauce
##
set -u # Throw error on uninitialized variables

for pkg in "${PYTHON_SRC_PACKAGES[@]}"
do
    cd "${PYTHON_SRC_PACKAGES_DIR}"/"${pkg}"
    pip install -e .
done
