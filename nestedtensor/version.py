__version__ = '0.0.1.dev2019122623+59ae23b'
git_version = '59ae23b086598d8775b0490fa9752a248bf0b11a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
