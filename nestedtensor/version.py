__version__ = '0.1.4+0759faf'
git_version = '0759fafed69d5571934a5a71d3226731ffe02429'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
