__version__ = '0.1.4+a3bff13'
git_version = 'a3bff1378e04f09983668e9a210ac2b73b06c41e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
