__version__ = '0.0.1.dev20208286+b1dd7b6'
git_version = 'b1dd7b692c45bcc4cee9f3408b30cb2722c5e271'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
