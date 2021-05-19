__version__ = '0.1.4+32c0be2'
git_version = '32c0be222686605de312e6ac67271308d18dfc71'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
