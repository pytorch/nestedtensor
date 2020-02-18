__version__ = '0.0.1.dev202021822+dd47d71'
git_version = 'dd47d71e2167e9e9227656be52dc52c71a2b9be8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
