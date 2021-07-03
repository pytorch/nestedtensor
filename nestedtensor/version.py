__version__ = '0.1.4+80a8bf2'
git_version = '80a8bf26fa88e9b5cf4e8b72e40a674ac2f78cff'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
