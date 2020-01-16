__version__ = '0.0.1.dev202011616+a4180bd'
git_version = 'a4180bdfda200de1244a4429bc668c10c0e1530c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
