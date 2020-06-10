__version__ = '0.0.1.dev20206102+58e4ad0'
git_version = '58e4ad0f1a3a5974ddc88b1d817bd059dc8e75fe'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
