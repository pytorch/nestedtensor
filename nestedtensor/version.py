__version__ = '0.1.4+de523be'
git_version = 'de523be1ca9bb799a8d3189c97c962b127f9b526'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
