__version__ = '0.0.1.dev20201111+d040bd1'
git_version = 'd040bd11803cae8469c4ef70090d34850ea6c615'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
