__version__ = '0.1.4+a34287d'
git_version = 'a34287d042096a3cc45adc085138b16d0dd6cb4a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
