__version__ = '0.0.1.dev202082620+4506e06'
git_version = '4506e063305dea953dd7740c931a0b2672970720'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
