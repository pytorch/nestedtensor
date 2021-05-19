__version__ = '0.1.4+aa4359a'
git_version = 'aa4359ac5ea5630c961631d66db67e2b902ab432'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
