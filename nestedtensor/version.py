__version__ = '0.0.1.dev202082819+36737a6'
git_version = '36737a6bdf48ea31427bf5e357c1f0cac84f14fb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
