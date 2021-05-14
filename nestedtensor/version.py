__version__ = '0.1.4+366d303'
git_version = '366d30342126474ecf4e45791836cda07ec6f473'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
