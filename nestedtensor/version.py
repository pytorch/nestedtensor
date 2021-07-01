__version__ = '0.1.4+660ea69'
git_version = '660ea69edd4e5ea9b9fa08e5721e761cde5d553e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
