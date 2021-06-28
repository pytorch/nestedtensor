__version__ = '0.1.4+a2bc19b'
git_version = 'a2bc19beffeb42eef647f702ff6062b1bd84f969'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
