__version__ = '0.1.4+a75bfab'
git_version = 'a75bfaba4bc1c08a8021f06c20f79186b9432dc5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
