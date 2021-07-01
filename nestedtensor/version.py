__version__ = '0.1.4+b6eadb3'
git_version = 'b6eadb3cbb3941577e07767d5555f020ae759450'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
