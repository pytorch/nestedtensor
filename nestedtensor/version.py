__version__ = '0.0.1.dev201912619+1c34b7e'
git_version = '1c34b7e151d4f04cb82450368aa36b66ff6a48a2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
