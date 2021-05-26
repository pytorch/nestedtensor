__version__ = '0.1.4+7782509'
git_version = '77825092a17cbd5c21eb737028183fa21adba9f8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
