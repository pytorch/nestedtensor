__version__ = '0.0.1.dev202082320+b218762'
git_version = 'b2187628c9fe16481c6db5208ed3e062a00ef61b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
