__version__ = '0.1.4+bf3132f'
git_version = 'bf3132f4fe9f6ef5f007cefba8a8a4c465a6bb72'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
