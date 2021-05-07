__version__ = '0.1.4+ac7ab07'
git_version = 'ac7ab07ff1b50081ce873f53754ae7f7854134d6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
