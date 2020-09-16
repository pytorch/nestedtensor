__version__ = '0.0.1.dev20209161+4e77225'
git_version = '4e772255cfdeb7049e60753e53774eb52fcded70'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
