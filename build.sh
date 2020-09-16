#!/bin/bash
pushd third_party/pytorch
python setup.py develop
popd
pip install -v -e .
