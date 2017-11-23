#!/usr/bin/env bash

set -e

COMPETITION=statoil-iceberg-classifier-challenge

# Get the file to submit
if [ $# -ne 1 ];
then
    echo 'You must provide a file to submit!'
    exit 1
fi
file=$1

if ! [ -f "$file" ]
then
    echo "Cannot find the file \`$file'!"
    exit 2
fi



# Input from user

echo -n 'Breif description of submission: '
read message

echo -n 'Username: '
read username

echo -n 'Password: '
read -s password
echo

# Do the submission
echo
echo 'Will now submit.'
echo 'If successful your score will be printed:'
kg submit "$file"           \
          -u "$username"    \
          -p "$password"    \
          -c "$COMPETITION" \
          -m "$message"
