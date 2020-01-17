__version__ = '0.0.1.dev202011622+5de3fd2'
git_version = '5de3fd2e150c10f99dea12a381f7db134f1308fe'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
