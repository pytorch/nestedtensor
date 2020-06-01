__version__ = '0.0.1.dev20206120+95f0d0c'
git_version = '95f0d0cc656ee718093e88445da9387069abbe29'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
