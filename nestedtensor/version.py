__version__ = '0.0.1.dev202082322+82225a8'
git_version = '82225a80f26b234f58303fb049b0795ca0b2ec0c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
