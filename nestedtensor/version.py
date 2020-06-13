__version__ = '0.0.1.dev20206136+e25392f'
git_version = 'e25392f4d73e9ccddebf37e077352e06c11526a2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
