__version__ = '0.0.1.dev202082323+82257eb'
git_version = '82257ebece6643de6f5ffe2f791a10240099c931'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
