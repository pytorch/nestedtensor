__version__ = '0.1.4+2004b53'
git_version = '2004b534364914fb593421f63468a1e710f91555'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
