__version__ = '0.1.4+2ccb26d'
git_version = '2ccb26dfa15130fc0b0ef1cadaa28fb6e2be187a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
