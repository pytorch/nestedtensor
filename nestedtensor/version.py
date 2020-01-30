__version__ = '0.0.1.dev20201301+e317493'
git_version = 'e3174933b22d2c9036cfe1be6735f617d1b052d6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
