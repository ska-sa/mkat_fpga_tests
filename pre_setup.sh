#!/bin/bash

set -e # Abort on any errors

PYTHON_SRC_PACKAGES=(katcp-python nosekatreport casperfpga corr2)

. venv/bin/activate

set -u # Throw error on uninitialized variables

for pkg in "${PYTHON_SRC_PACKAGES[@]}"
do
    cd "${PYTHON_SRC_PACKAGES_DIR}"/"${pkg}"
    pip install -e .
done
