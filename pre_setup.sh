#!/bin/bash

set -e # Abort on any errors
set -u # Throw error on uninitialized variables

PYTHON_SRC_PACKAGES=(corr2 casperpga katcp-python)

. venv/bin/activate

for pkg in "${PYTHON_SRC_PACKAGES[@]}"
do
    cd $PYTHON_SRC_PACKAGES_DIR/$pkg
    pip install -e .
done
