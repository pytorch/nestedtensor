__version__ = '0.0.1.dev201912271+9e96318'
git_version = '9e96318f047603470e68e268ec3a0ffa27410262'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
