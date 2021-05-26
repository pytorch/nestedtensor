__version__ = '0.1.4+3967c33'
git_version = '3967c33b1f7819174032c6c2bba73532336a0fa2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
