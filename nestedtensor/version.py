__version__ = '0.0.1.dev202022121+18bfa86'
git_version = '18bfa86636097fe382d96977814b3a751e7396f3'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
