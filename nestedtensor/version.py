__version__ = '0.0.1.dev202022321+aead1bd'
git_version = 'aead1bdc2542ab0e34c087c01e52e87b61ec7be9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
