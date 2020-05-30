__version__ = '0.0.1.dev202053019+997d15a'
git_version = '997d15a6d611a042cfba84079b7b1be31b90a595'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
