__version__ = '0.0.1.dev20204251+f658ddf'
git_version = 'f658ddf05aab845b51bc8c055b141d9fb63483af'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
