__version__ = '0.0.1.dev20201111+ca117cc'
git_version = 'ca117cc21eae925a93b343a3e5beeca843ff35cc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
