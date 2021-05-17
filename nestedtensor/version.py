__version__ = '0.1.4+6c54e9e'
git_version = '6c54e9e3d70c798de6839bf9345c15072a37b916'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
