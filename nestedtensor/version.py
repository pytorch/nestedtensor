__version__ = '0.0.1.dev20207192+2cb6c46'
git_version = '2cb6c4667ba2ed5897cf3330a8dc93aef9445563'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
