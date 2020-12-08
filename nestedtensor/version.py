__version__ = '0.0.1+097a691'
git_version = '097a6919be15b30fb02ec4eb957071e5d08d13ab'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
