__version__ = '0.1.4+e11c455'
git_version = 'e11c455b836bceff09da61ec7610ed39b72dc52b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
