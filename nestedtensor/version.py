__version__ = '0.0.1.dev20203322+bbc52d0'
git_version = 'bbc52d0d9ca1b2f49cb93016caf5cfbee66a2628'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
