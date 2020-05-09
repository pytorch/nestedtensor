__version__ = '0.0.1.dev2020594+15b71a6'
git_version = '15b71a658b9e5b633223525d075fbc4fd7a3b04d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
