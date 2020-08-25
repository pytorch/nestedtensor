__version__ = '0.0.1.dev20208254+bfefe75'
git_version = 'bfefe757d4cbfd7acc7303497980ea9816c8e5f4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
