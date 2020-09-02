__version__ = '0.0.1.dev2020921+9683c8f'
git_version = '9683c8f30e804a8dfb7bf68f61a4ee66428c0e40'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
