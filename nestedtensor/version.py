__version__ = '0.0.1.dev20202221+0986370'
git_version = '0986370cf19eb2a32ceb0d58e57b766decfa48bf'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
