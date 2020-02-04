__version__ = '0.0.1.dev20202418+78d5655'
git_version = '78d5655fdda52346b65bc272d562800a298b2d07'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
