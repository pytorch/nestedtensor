__version__ = '0.1.4+a5e6cd0'
git_version = 'a5e6cd0651d42bb1968a0da56974e3b5bf4e758c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
