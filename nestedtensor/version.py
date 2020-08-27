__version__ = '0.0.1.dev202082721+9a173e6'
git_version = '9a173e6c65e89a9b74611fc98002ef1e0be385da'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
