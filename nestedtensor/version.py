__version__ = '0.0.1.dev20202136+ac0cf98'
git_version = 'ac0cf98c01ed1a1c7f22f90a127a6fd9bd4726cb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
