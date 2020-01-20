__version__ = '0.0.1.dev20201207+73f281d'
git_version = '73f281dfdf292cd22926a784c6f899a0e02ab62b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
