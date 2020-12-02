__version__ = '0.0.1.dev202012222+8b05de3'
git_version = '8b05de32a0acdbfeedade06162dac5d42a73d082'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
