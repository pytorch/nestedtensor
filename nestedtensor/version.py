__version__ = '0.0.1+9627476'
git_version = '9627476a0be0a32e628d6c63bf772a1483926613'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
