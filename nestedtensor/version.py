__version__ = '0.1.4+8629023'
git_version = '862902387357f280fba58037898b735b1742f207'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
