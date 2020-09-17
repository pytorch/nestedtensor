#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

if [ "${CU_VERSION:-}" == cpu ] ; then
    cudatoolkit="cpuonly"
else
    if [[ ${#CU_VERSION} -eq 4 ]]; then
        CUDA_VERSION="${CU_VERSION:2:1}.${CU_VERSION:3:1}"
    elif [[ ${#CU_VERSION} -eq 5 ]]; then
        CUDA_VERSION="${CU_VERSION:2:2}.${CU_VERSION:4:1}"
    fi
    echo "Using CUDA $CUDA_VERSION as determined by CU_VERSION"
    version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"
    cudatoolkit="cudatoolkit=${version}"
fi

printf "Checking out submodules for pytorch build\n"
git submodule sync
git submodule update --init --recursive
conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
printf "* Installing NT-specific pytorch and nestedtensor\n"
USE_DISTRIBUTED=OFF BUILD_TEST=OFF BUILD_CAFFE2_OPS=0 USE_NUMPY=ON USE_NINJA=1 ./clean_build.sh

printf "* Installing torchvision from source for testing\n"
rm -rf /tmp/vision
git clone https://github.com/pytorch/vision /tmp/vision

pushd /tmp/vision
python setup.py clean
python setup.py develop
popd
