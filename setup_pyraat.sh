#!/bin/bash
# Make sure python will be able to find the installed shared libraries
# Can be run from any directory. Not necessary to be in directory of this script
PYTHONPATH=/usr/local/PyRAAT/
PYTHONPATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )/python/:$PYTHONPATH
#TODO: Decide if we should add other python modules such as primitives
#PYTHONPATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )/python/primitives/:$PYTHONPATH

export PYTHONPATH
