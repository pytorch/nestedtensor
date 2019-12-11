__version__ = '0.0.1.dev2019121123+bef337d'
git_version = 'bef337dc4588932ef0dc2486c75d2f3d0cb4533d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
