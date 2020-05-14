__version__ = '0.0.1.dev202051421+159974e'
git_version = '159974e38ea51f28bff88fd861ae3c3573706691'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
