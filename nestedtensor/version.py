__version__ = '0.1.4+a2a84cf'
git_version = 'a2a84cfda1b906a1a0201e5f3e447c38e8b2e3bb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
