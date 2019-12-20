__version__ = '0.0.1.dev201912204+c9b2bef'
git_version = 'c9b2bef345cde60912cc9d78ef46b96c54e8fb3e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
