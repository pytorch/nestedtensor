__version__ = '0.0.1.dev202071515+0afa049'
git_version = '0afa049afe7df5c7b4076b66acf4f4d185bbeff3'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
