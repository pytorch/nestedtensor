# Expects cuda 10.2 environment

WHEELS_FOLDER=${HOME}/project/wheels
mkdir -p $WHEELS_FOLDER
PYTHON_VERSION="3.8"
PYTORCH_VERSION=""
UNICODE_ABI=""
CU_VERSION="cu101"
python setup.py clean
DEBUG=0 USE_NINJA=1 python setup.py develop bdist_wheel -d $WHEELS_FOLDER
