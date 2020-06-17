__version__ = '0.0.1.dev20206174+ffc272c'
git_version = 'ffc272cceeccc5b88ff0620d11a499c9da4fd269'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
