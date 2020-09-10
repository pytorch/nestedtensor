__version__ = '0.0.1.dev20209104+164a27a'
git_version = '164a27a41d5a500ab194a092509e21212e4721e7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
