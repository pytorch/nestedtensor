__version__ = '0.1.4+601ecf3'
git_version = '601ecf30b0b7569c7daf041bafe581e034b5bd4a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
