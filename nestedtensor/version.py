__version__ = '0.0.1.dev202092921+e0a8e87'
git_version = 'e0a8e8793a3ea3bdca4ccec18346d6d2526e4a3c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
