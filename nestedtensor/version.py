__version__ = '0.1.4+7dcd033'
git_version = '7dcd03339b224161e469799fc79530650e9af137'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
