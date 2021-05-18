#!/usr/bin/env bash

# Expects cuda 10.2 environment

WHEELS_FOLDER=${HOME}/project/wheels
mkdir -p $WHEELS_FOLDER
python setup.py clean
PYTHON_VERSION="3.7" PYTORCH_VERSION="" UNICODE_ABI="" CU_VERSION="cpu" BUILD_VERSION="0.1.5.dev20210429" DEBUG=0 USE_NINJA=1 python setup.py develop bdist_wheel -d $WHEELS_FOLDER
