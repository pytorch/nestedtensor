__version__ = '0.0.1.dev20204254+bb10d71'
git_version = 'bb10d71cd9fcbbe5bbde115a5f0c3e051ea814f0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
