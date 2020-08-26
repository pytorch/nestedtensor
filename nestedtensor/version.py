__version__ = '0.0.1.dev202082621+8c8f92e'
git_version = '8c8f92e98b5cf8f860b514bd49f0cc1457237874'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
