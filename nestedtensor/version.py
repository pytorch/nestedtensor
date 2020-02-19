__version__ = '0.0.1.dev20202194+0945ed0'
git_version = '0945ed06e9b62ebac42107a3341e833d7ea50be9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
