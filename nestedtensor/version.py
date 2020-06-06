__version__ = '0.0.1.dev2020665+ec3f819'
git_version = 'ec3f8198fbd9396a251808930ccb62d697fdca38'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
