__version__ = '0.1.4+bcca124'
git_version = 'bcca12440771be1f3727be1a46f12cef3659e662'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
