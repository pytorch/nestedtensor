__version__ = '0.0.1.dev20208316+f75ebd4'
git_version = 'f75ebd44dc72675a80e7c7ee72c99c30203387f2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
