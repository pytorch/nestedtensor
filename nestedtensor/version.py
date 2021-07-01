__version__ = '0.1.4+f7cfcbe'
git_version = 'f7cfcbe6dea564df7bc1ebe81e9908d701531f18'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
