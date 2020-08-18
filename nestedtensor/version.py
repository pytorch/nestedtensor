__version__ = '0.0.1.dev20208180+e2ae49a'
git_version = 'e2ae49a89c15c7f24a7fe9eac7ea29aae5e3b130'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
