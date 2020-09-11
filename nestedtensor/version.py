__version__ = '0.0.1.dev202091019+76e54ed'
git_version = '76e54ed6e91f8e022f3fad5309a03325ae0949c4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
