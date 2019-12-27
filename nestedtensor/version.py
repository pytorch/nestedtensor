__version__ = '0.0.1.dev201912272+64c437c'
git_version = '64c437c9e5a450ede231a6de6901ef46c2b22acb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
