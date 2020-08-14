__version__ = '0.0.1.dev20208144+277b06b'
git_version = '277b06bb8d11f3e75aee8dba48cb5ed25411eaf4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
