__version__ = '0.0.1.dev2019122721+7ce1e24'
git_version = '7ce1e243f4f622c674fce0dbae37d3b0f9b2a34f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
