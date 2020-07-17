__version__ = '0.0.1.dev202071721+2c7762e'
git_version = '2c7762ebe9884dd69c10880ec5b9f04234d3d2a8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
