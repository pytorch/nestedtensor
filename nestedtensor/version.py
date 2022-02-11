__version__ = '0.1.4+244aecd'
git_version = '244aecdcc9389bf6ab462248164fb62b4b018ac1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
