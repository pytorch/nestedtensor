__version__ = '0.0.1.dev20191281+36b1800'
git_version = '36b1800b8416160ba355ff5b2eab01d629d4258a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
