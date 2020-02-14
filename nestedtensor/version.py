__version__ = '0.0.1.dev202021421+9410f4a'
git_version = '9410f4a5b8815130cfae73768d886ab1823fa9ce'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
