__version__ = '0.1.4+7db1ebb'
git_version = '7db1ebbc51501749b19f92c7f0e04ecb0bff7528'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
