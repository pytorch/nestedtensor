#!/bin/bash
set -e
set -x
pushd third_party/pytorch
python setup.py develop
popd
pip install -v -e .
