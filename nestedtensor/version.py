__version__ = '0.0.1.dev202071015+cc878b4'
git_version = 'cc878b483503bd0787a998414c08ab54f1333260'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
