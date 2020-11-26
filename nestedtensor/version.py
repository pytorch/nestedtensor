__version__ = '0.0.1.dev202011260+3ea47fa'
git_version = '3ea47fa199ec755ee004b54bd1ae94422c9cfdf5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
