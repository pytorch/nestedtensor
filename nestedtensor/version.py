__version__ = '0.0.1.dev202031220+0b18247'
git_version = '0b182471c373874a5def9f1c54b1c74fa4e617d6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
