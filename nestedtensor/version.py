__version__ = '0.1.4+8de8f64'
git_version = '8de8f6487f58d0e17615f125bd0049b3ef16f2dc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
