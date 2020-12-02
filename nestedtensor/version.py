__version__ = '0.0.1+ae63be8'
git_version = 'ae63be897b86b23a39bfdd01f79ad088467812fc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
