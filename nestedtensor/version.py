__version__ = '0.0.1.dev20208215+62e38fd'
git_version = '62e38fd7fb2f09d5754aa4881c2d90df70ec6401'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
