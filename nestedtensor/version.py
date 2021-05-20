__version__ = '0.1.4+5122e9e'
git_version = '5122e9e33b481abebfccf9f0702885958a2bc791'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
