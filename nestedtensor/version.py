__version__ = '0.1.4+22fbb73'
git_version = '22fbb731ffbf982ed016795e3be6c19f22e15e58'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
