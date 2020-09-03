__version__ = '0.0.1.dev2020937+82646d2'
git_version = '82646d2decd2acc7a0724350553401137466b3fa'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
