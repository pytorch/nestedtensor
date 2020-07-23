__version__ = '0.0.1.dev202072317+92a8cf1'
git_version = '92a8cf19c0108698b6b09a9565c15a183d711884'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
