__version__ = '0.0.1.dev20208235+8b2d0dc'
git_version = '8b2d0dc79cbc9ade6299eb5e417d49b6f04822f2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
