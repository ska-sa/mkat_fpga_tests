#!/bin/bash

set -e # Abort on any errors
set -u # Throw error on uninitialized variables

SRCDIR=$1			# Where source dependency repos are
PYTHON_SRC_PACKAGES=(corr2 casperpga katcp-python)

. venv/bin/activate

for pkg in "${PYTHON_SRC_PACKAGES[@]}"
do
    cd $SRCDIR/$pkg
    pip install -e .
done



