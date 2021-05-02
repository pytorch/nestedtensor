__version__ = '0.1.4+4df33eb'
git_version = '4df33ebdfab681e907c5b65e8d154ea4c2a2289a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
