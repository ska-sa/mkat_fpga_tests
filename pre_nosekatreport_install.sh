#!/bin/bash

set -e # Abort on any errors

. venv/bin/activate

nosepath=$(pwd)
echo 'Installing nosekatreport: ' "$nosepath"/nosekatreport
cd "$nosepath"/nosekatreport
pip install --force-reinstall -e .

set -u # Throw error on uninitialized variables
