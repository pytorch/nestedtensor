__version__ = '0.1.4+7df079d'
git_version = '7df079de1aae06f4603f2616c14ef4188442c75e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
