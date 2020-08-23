__version__ = '0.0.1.dev20208234+59aaae0'
git_version = '59aaae0aa20f51bf13d43919abfc1c904b14f03a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
