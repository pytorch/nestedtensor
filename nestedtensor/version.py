__version__ = '0.0.1.dev20207163+2e502ba'
git_version = '2e502ba66ffd1e5ec4548734eb7c91ff4c8af683'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
