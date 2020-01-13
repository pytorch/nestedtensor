__version__ = '0.0.1.dev202011321+221f332'
git_version = '221f33287feee5f7225d70bf1f13e57f1609bcc0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
