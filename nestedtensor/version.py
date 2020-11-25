__version__ = '0.0.1.dev2020112523+e7bb40b'
git_version = 'e7bb40ba8c00d8e6a305f6b3e67a01ead0fa4372'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
