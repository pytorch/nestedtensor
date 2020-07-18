__version__ = '0.0.1.dev20207183+57adbbb'
git_version = '57adbbb2d2a5080b089700f840f979437f2ca1e0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
