__version__ = '0.1.4+e26fd8e'
git_version = 'e26fd8ecaf0ed8f9a8f0b8424f27516fc48e7384'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
