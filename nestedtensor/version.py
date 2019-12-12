__version__ = '0.0.1.dev2019121223+4b136fa'
git_version = '4b136fa49a4b3b010195145936415a265406ae76'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
