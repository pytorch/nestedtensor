__version__ = '0.0.1.dev20208294+5d3620d'
git_version = '5d3620da32c8fe6a7983129c9ba2eaded03fe209'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
