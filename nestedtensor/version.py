__version__ = '0.1.4+ef34899'
git_version = 'ef348991e29efa4d9670ac21f0c98bfec3af98c8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
