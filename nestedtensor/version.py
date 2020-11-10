__version__ = '0.0.1.dev202011103+795eeea'
git_version = '795eeea4c8846dab3eea70e61c2337829c54952c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
