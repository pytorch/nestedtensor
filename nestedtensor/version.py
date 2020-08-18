__version__ = '0.0.1.dev20208185+63c39c7'
git_version = '63c39c714c7b52330f460a016d414578e7e80de0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
