__version__ = '0.0.1.dev202072822+e224ff4'
git_version = 'e224ff47ec2fd64ad4aa23bc9c70511361633e42'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
