__version__ = '0.1.4+af85059'
git_version = 'af8505983f65eb7d54c02126b8ba26c26f1505eb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
