__version__ = '0.1.4+e2aaac9'
git_version = 'e2aaac95b3387e10d62fc5777e27ae32b4c27b07'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
