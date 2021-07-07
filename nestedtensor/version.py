__version__ = '0.1.4+43c7eb4'
git_version = '43c7eb4ecefdce41b208c1c9e9a3c6d2ddbbe02b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
