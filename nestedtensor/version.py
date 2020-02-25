__version__ = '0.0.1.dev20202252+6303438'
git_version = '630343863d50df752a6bd92c1da1663a36ecea56'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
