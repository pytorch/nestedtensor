__version__ = '0.0.1.dev20202422+a901708'
git_version = 'a901708fe4d9095a9d5391f6882394aca4453b68'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
