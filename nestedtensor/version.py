__version__ = '0.0.1.dev202083118+b636f39'
git_version = 'b636f39ea12d97ea7462dda330985f1f9b572da4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
