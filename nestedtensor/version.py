__version__ = '0.0.1+c4f56a4'
git_version = 'c4f56a4aac0b24f18b9ed3875f0c340982f679af'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
USE_SUBMODULE=False
