__version__ = '0.0.1.dev202010233+95059d9'
git_version = '95059d9166377471809a528dd756321c296b08c7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
