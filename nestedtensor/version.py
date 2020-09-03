__version__ = '0.0.1.dev20209219+88d1a7e'
git_version = '88d1a7eb8b1f4f8b097fcc89c7ad36809cd05035'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
