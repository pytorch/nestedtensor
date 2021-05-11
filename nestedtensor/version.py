__version__ = '0.1.4+a8aa838'
git_version = 'a8aa8381ba60e840e5503a45ddcfd6668422a285'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
