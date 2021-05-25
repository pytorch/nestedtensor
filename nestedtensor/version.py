__version__ = '0.1.4+27882c6'
git_version = '27882c62d6961f0e39875ef34bb27cfb653a22b7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
