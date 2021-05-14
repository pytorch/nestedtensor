__version__ = '0.1.4+d2c4034'
git_version = 'd2c4034237441f6d4dd4de02678bdba55136639f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
