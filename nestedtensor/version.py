__version__ = '0.1.4+01575e6'
git_version = '01575e65b91f313cd976bc36f05d1f3d092ebf48'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
