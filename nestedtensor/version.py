__version__ = '0.1.4+6f5f025'
git_version = '6f5f02513f85bfa9cf0642555a02507701d18d57'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
