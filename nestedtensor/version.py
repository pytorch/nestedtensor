__version__ = '0.0.1.dev202021422+c9ebe3f'
git_version = 'c9ebe3f63461dce5cd92195d5bb0f20e68ceae07'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
