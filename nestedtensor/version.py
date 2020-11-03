__version__ = '0.0.1.dev202011321+771d63c'
git_version = '771d63cb4e14d2b6f31892f59abb6068eb5e5ae7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
