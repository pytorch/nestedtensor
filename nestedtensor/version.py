__version__ = '0.0.1.dev20201172+5685bfc'
git_version = '5685bfc7ea7ae970c3d71cce5e8cd2f57ae8d3dd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
