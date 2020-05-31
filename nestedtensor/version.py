__version__ = '0.0.1.dev202053121+5f42d91'
git_version = '5f42d913f8c0acc496be1ddb742e44e881b3f3ed'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
