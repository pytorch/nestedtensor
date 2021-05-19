__version__ = '0.1.4+2e9ebd4'
git_version = '2e9ebd435c0ff0065667480c0fdbf97aa416df61'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
