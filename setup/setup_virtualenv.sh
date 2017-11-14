#!/usr/bin/env bash

set -e

virtualenv env -p python3

if [ -f env/bin/activate ]; then
    source env/bin/activate
elif [ -f env/Scripts/activate ]; then
    source env/Scripts/activate
else
    echo "Unable to activate the python virtual environment!"
    exit 1
fi

pip install -r setup/REQUIREMENTS.txt
