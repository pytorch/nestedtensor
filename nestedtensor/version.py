__version__ = '0.0.1.dev202061718+94aadd5'
git_version = '94aadd589c657f116508eb3547104b19ddbcb63c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
