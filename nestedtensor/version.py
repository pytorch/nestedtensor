__version__ = '0.0.1.dev20208622+d1ab507'
git_version = 'd1ab507267fd0eb57f32454431ce6d8e7965026a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
