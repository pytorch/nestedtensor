__version__ = '0.0.1.dev20202422+439e1e8'
git_version = '439e1e8fdb661b69109e24d7a468b247af0d0ff2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
