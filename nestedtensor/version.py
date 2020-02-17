__version__ = '0.0.1.dev20202174+b488897'
git_version = 'b4888970306b1f0fc1f1e70f08273a7bc7a2881e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
