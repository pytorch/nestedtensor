__version__ = '0.0.1.dev202032320+857f162'
git_version = '857f162bfd95d512ff6816633883eeaf0fd3115f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
