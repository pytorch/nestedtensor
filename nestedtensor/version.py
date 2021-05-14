__version__ = '0.1.4+393af27'
git_version = '393af2758859ac3092a3544fe234646f91501c62'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
