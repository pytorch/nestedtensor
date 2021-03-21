__version__ = '0.0.1+9a1469b'
git_version = '9a1469b697960263f92e654c16682882996bcd0b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
