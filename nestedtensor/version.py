__version__ = '0.1.4+87981a2'
git_version = '87981a22d96f271754989031475491bda355a215'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
