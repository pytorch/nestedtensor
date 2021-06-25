__version__ = '0.1.4+7b09dfe'
git_version = '7b09dfef0fd3f22475515072f1df362eb1c84edc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
