__version__ = '0.0.1.dev202082018+24f6849'
git_version = '24f6849c4e3e1668d875e50486e2530cf3e572cb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
