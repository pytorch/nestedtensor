__version__ = '0.0.1.dev20208315+177d1e7'
git_version = '177d1e77164161a442fbe2cff25872b69732b639'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
