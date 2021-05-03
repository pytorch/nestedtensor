__version__ = '0.1.4+158fcde'
git_version = '158fcdef9a6bcc7638ffa319e4e8efdeb7f0e495'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
