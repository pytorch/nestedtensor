__version__ = '0.0.1.dev20208264+7ec5a74'
git_version = '7ec5a742517bb211fc905c8e6e610174e712debe'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
