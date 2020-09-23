__version__ = '0.0.1.dev202092322+f3d1a75'
git_version = 'f3d1a759ba4ea0eb1466947342ca249b116d6aba'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
