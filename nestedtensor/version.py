__version__ = '0.0.1.dev201912271+ef00f30'
git_version = 'ef00f309f2eb521ef6950f8c4b23f0b51721b132'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
