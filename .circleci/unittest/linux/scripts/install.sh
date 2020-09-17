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

# # torchvision is used for testing only
# printf "Installing PyTorch and torchvision with %s\n" "${cudatoolkit}"
# conda install -y -c pytorch-nightly pytorch torchvision "${cudatoolkit}"

# python setup.py develop

# USE_DISTRIBUTED=OFF BUILD_TEST=OFF BUILD_CAFFE2_OPS=0 USE_FBGEMM=OFF USE_NUMPY=ON USE_QNNPACK=OFF USE_PYTORCH_QNNPACK=OFF USE_CUDA=OFF USE_XNNPACK=OFF USE_NNPACK=OFF DEBUG=1 USE_NINJA=1 ./clean_build.sh
printf "* Installing NT-specific pytorch and nestedtensor\n"
./clean_build.sh

printf "* Installing torchvision from source for testing\n"
rm -rf /tmp/vision
git clone https://github.com/pytorch/vision /tmp/vision

pushd /tmp/vision
pip install --upgrade pip
pip install numpy pyyaml future
pip install ninja
python setup.py clean
python setup.py develop
popd
