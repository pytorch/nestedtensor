#!/bin/bash
set -e
set -x
pushd third_party/pytorch
python setup.py clean
python setup.py develop
popd
python setup.py clean
python setup.py develop
