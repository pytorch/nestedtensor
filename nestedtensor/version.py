__version__ = '0.0.1+9ad1d6f'
git_version = '9ad1d6f78df323d04eef188eef909ca1190027a5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
