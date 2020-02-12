__version__ = '0.0.1.dev202021222+c7950ed'
git_version = 'c7950ed5a0746af71cc9b84ddcbffa2f1ba0815c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
