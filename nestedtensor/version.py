__version__ = '0.0.1.dev20205151+a913257'
git_version = 'a91325744864edce8aee590a422c6f958f13e889'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
