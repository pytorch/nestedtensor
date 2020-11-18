__version__ = '0.0.1.dev202011182+86517e8'
git_version = '86517e844d093c7b0e142988b1a21e078c64eaf6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
