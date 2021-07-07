__version__ = '0.1.4+ec86343'
git_version = 'ec863433e19b6b0915e736fbbaa292529626a87e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
