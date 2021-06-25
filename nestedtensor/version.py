__version__ = '0.1.4+ac3cfd0'
git_version = 'ac3cfd027ff1518cfbaa94a383fb5da848220d18'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
