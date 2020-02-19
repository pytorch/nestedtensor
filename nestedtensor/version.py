__version__ = '0.0.1.dev20202196+98d4291'
git_version = '98d4291137828b3559c4373b7fe0bb727ba2d211'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
