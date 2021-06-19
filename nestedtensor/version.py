__version__ = '0.1.4+ceedfa3'
git_version = 'ceedfa36e61985d4931e401733b5eb895a0789ca'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
