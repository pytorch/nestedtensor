__version__ = '0.1.4+d75223e'
git_version = 'd75223e769a9ff08e2e1d4dd8df8d7c67a2caf8a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
