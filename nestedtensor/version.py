__version__ = '0.1.4+76b74c0'
git_version = '76b74c0e0ecc9ead1b64c2bc31f577e78b87dd00'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
