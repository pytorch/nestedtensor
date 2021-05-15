__version__ = '0.1.4+5a3c250'
git_version = '5a3c250f8f1d9430ce8a67c6ab3ffcabb054a9a5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
