__version__ = '0.0.1.dev20201205+f6d9321'
git_version = 'f6d9321b1bbc99769153756b3a66bf18b2e5ba19'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
