__version__ = '0.0.1.dev20204251+8f703e7'
git_version = '8f703e7e7f3d6838fb1b1ea06346f59bcc780ba4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
