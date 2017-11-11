#!/usr/bin/env bash

set -e

virtualenv env -p python3

source env/bin/activate

pip install -r setup/REQUIREMENTS.txt
