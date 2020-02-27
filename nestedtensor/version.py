__version__ = '0.0.1.dev20202255+567b4f8'
git_version = '567b4f869f6d676d77ec5fcea0a854a980d54334'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
