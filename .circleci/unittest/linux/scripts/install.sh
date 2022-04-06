#!/usr/bin/env bash
set -x
set -e

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

PYVSHORT=${PARAMETERS_PYTHON_VERSION:0:1}${PARAMETERS_PYTHON_VERSION:2:1}

if [[ "$PYVSHORT" == "38" ]] ; then
   PYVSHORT=cp${PYVSHORT}-cp${PYVSHORT}
elif [[ "$PYVSHORT" == "39" ]] ; then
   PYVSHORT=cp${PYVSHORT}-cp${PYVSHORT}
else
   PYVSHORT=cp${PYVSHORT}-cp${PYVSHORT}m
fi

NIGHTLY_DATE=20220224

if [ "${CU_VERSION:-}" == cpu ] ; then
    pip3 install -q --pre torch==1.12.0dev${NIGHTLY_DATE} torchvision==0.13.0dev${NIGHTLY_DATE}+cpu -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
    conda install -y ninja
    PYTORCH_VERSION="$(python -c "import torch; print(torch.__version__)")" USE_NINJA=1 python setup.py develop bdist_wheel -d $WHEELS_FOLDER
else
    pip3 install -q --pre torch==1.12.0dev${NIGHTLY_DATE}+cu111 torchvision==0.13.0dev${NIGHTLY_DATE} -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html
    conda install -y ninja
    PYTORCH_VERSION="$(python -c "import torch; print(torch.__version__)")" FORCE_CUDA=1 USE_NINJA=1 python setup.py develop bdist_wheel -d $WHEELS_FOLDER
fi

# if [ "${CU_VERSION:-}" == cpu ] ; then
#     conda install -y pytorch torchvision cpuonly -c pytorch-nightly
#     PYTORCH_VERSION="$(python -c "import torch; print(torch.__version__)")" USE_NINJA=1 python setup.py develop bdist_wheel -d $WHEELS_FOLDER
# else
#     conda install -y pytorch torchvision cudatoolkit=10.2 -c pytorch-nightly
#     PYTORCH_VERSION="$(python -c "import torch; print(torch.__version__)")" FORCE_CUDA=1 USE_NINJA=1 python setup.py develop bdist_wheel -d $WHEELS_FOLDER
# fi
