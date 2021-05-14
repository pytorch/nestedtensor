__version__ = '0.1.4+678d2ae'
git_version = '678d2aebd2994e1d9f9b9b8cd3ec2b8ad83ded0e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
