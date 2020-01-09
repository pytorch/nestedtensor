__version__ = '0.0.1.dev20201919+d20dbe9'
git_version = 'd20dbe9111f682dca1aef880ea380317b2e2e8fe'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
