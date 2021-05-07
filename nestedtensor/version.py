__version__ = '0.1.4+67cf479'
git_version = '67cf479144e85f966ee0c189ac411f049e84b426'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
