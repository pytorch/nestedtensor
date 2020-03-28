__version__ = '0.0.1.dev20203284+6ddc5c4'
git_version = '6ddc5c4da724206bf0f3d2c2bcf1552eb3f20386'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
