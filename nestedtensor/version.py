__version__ = '0.1.4+2d3eb86'
git_version = '2d3eb8610381cd1785de853fee6c27a164b975ea'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
