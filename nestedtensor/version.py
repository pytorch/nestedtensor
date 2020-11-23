__version__ = '0.0.1.dev2020112316+12d4be3'
git_version = '12d4be347fb47542c508a26edeeb227be6353f1b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
