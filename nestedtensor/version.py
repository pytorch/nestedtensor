__version__ = '0.0.1.dev20202250+c8a7fd4'
git_version = 'c8a7fd435605b34d8affe2c385ff07ba6988b730'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
