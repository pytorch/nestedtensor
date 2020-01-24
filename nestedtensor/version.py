__version__ = '0.0.1.dev202012420+078f5ef'
git_version = '078f5ef72834906d846179111c34eacb653d184d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
