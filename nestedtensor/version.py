__version__ = '0.0.1.dev202073118+69777a3'
git_version = '69777a3db82b4f10d1cf8ff5e84db8d4d7f6d6e1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
