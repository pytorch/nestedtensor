#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

# if [ "${CU_VERSION:-}" == cpu ] ; then
#     cudatoolkit="cpuonly"
# else
#     if [[ ${#CU_VERSION} -eq 4 ]]; then
#         CUDA_VERSION="${CU_VERSION:2:1}.${CU_VERSION:3:1}"
#     elif [[ ${#CU_VERSION} -eq 5 ]]; then
#         CUDA_VERSION="${CU_VERSION:2:2}.${CU_VERSION:4:1}"
#     fi
#     echo "Using CUDA $CUDA_VERSION as determined by CU_VERSION"
#     version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"
#     cudatoolkit="cudatoolkit=${version}"
# fi

WHEELS_FOLDER=${HOME}/project/wheels
mkdir -p $WHEELS_FOLDER

if [ "${CU_VERSION:-}" == cpu ] ; then
    conda install -y pytorch torchvision cpuonly -c pytorch-nightly
else
    conda install -y pytorch torchvision cudatoolkit=10.2 -c pytorch-nightly
fi
