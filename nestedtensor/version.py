__version__ = '0.0.1.dev20203322+5a61955'
git_version = '5a6195554548a4b09041cb59abe45ac5d0e79f4a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
