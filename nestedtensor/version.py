__version__ = '0.0.1.dev2019122620+82b0f0f'
git_version = '82b0f0f0c6ffaac1e337b21156f3efbcffcead19'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
