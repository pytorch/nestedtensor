__version__ = '0.0.1.dev202012819+4b27d69'
git_version = '4b27d69dfbf226335f61689103d808d79e277a73'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
