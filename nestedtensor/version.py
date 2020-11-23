__version__ = '0.0.1.dev2020112316+674ff9b'
git_version = '674ff9bd037f5fc77e70ec21736bf85bdb60f0ca'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
