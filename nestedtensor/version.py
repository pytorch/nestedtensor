__version__ = '0.0.1.dev202091016+9a0f5f7'
git_version = '9a0f5f7c27d011f01b8e78ec9a1537f47e9e7a1a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
