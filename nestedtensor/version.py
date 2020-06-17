__version__ = '0.0.1.dev20206174+332e611'
git_version = '332e6110d3be0120a5073f18f6658a9e4b5fb0b5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
