__version__ = '0.0.1.dev20202174+9cb2194'
git_version = '9cb21941e603e8debc173a319e615f94ce76d0a7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
