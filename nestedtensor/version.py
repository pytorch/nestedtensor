__version__ = '0.0.1.dev20208293+e2b5042'
git_version = 'e2b50422b09124941595b59325fc27bd5fc13df8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
