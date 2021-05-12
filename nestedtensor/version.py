__version__ = '0.1.4+aa42a84'
git_version = 'aa42a84ed28c9c4e9aa838b456e9101c26c49370'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
