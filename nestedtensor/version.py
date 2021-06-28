__version__ = '0.1.4+bc0a848'
git_version = 'bc0a8485884bd2e0f58946689343d8ee45229486'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
