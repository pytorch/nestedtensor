__version__ = '0.0.1+ff6e79b'
git_version = 'ff6e79b822223e2de095fbb63124ec06f164455a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
