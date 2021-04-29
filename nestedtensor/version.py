__version__ = '0.1.4dev20210429'
git_version = 'ad60f981216ae161e3927e3fbe9c086c5c38ffa0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
