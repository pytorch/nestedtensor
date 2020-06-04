__version__ = '0.0.1.dev2020644+ebbfafb'
git_version = 'ebbfafbd1b03c746cbddd627ccfffd3b4a994652'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
