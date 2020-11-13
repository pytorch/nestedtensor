__version__ = '0.0.1.dev202011131+5485b6d'
git_version = '5485b6d6f52e92c0241ef079dd072616d8e8dba3'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
