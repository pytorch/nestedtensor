__version__ = '0.1.4+c95d319'
git_version = 'c95d31944c0693c34d58802d677c7e92be81eb18'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
