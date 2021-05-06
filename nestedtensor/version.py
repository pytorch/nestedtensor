__version__ = '0.1.4+7f199f4'
git_version = '7f199f49bb6e4fff89f9f6b2cbb7e7a32518c4ba'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
