__version__ = '0.1.4+be38ffb'
git_version = 'be38ffbe083cea3682167c52bb14617704324316'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
