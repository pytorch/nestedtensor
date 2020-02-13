__version__ = '0.0.1.dev20202132+acba369'
git_version = 'acba36992bd95a2627792e590cf5db2439af6cdf'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
