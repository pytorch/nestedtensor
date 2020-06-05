__version__ = '0.0.1.dev20206521+c4f0723'
git_version = 'c4f072382d3926a61018d5d0d846bf72cb484c49'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
