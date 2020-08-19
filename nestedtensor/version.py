__version__ = '0.0.1.dev20208191+54881e8'
git_version = '54881e8526ca426ce0b24ff5932b22e9efdcb2a4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
