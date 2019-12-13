__version__ = '0.0.1.dev2019121321+baa3c0c'
git_version = 'baa3c0cd041f0969e5af41a3f1a9ac876e7cb3af'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
