__version__ = '0.0.1.dev20208302+e41ffb8'
git_version = 'e41ffb86a682361954d812ca9f97c762987f4333'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
