__version__ = '0.0.1.dev20202194+fbec8e4'
git_version = 'fbec8e48f6eda663a0a28805932db1c6a10d19c4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
