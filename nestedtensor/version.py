__version__ = '0.0.1+ccb9a16'
git_version = 'ccb9a164d30f57997f7fde6edb9dbd35bb401676'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
