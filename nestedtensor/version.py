__version__ = '0.1.4+58f6a9b'
git_version = '58f6a9bc4e8e9fe0406f58af4f0e541a553c1997'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
