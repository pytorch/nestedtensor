__version__ = '0.1.4+37cd5db'
git_version = '37cd5db5161446e66cd3c2eb5fc1e4299b57b11a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
