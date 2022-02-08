__version__ = '0.1.4+abd0ac3'
git_version = 'abd0ac3fcbeaed1243bc116fc297beac2ec3c29b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
