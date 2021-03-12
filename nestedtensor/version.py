__version__ = '0.0.1+e409f00'
git_version = 'e409f00f6a17f4c162117c73cc32a41b76acf692'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
USE_C_EXTENSION=True
