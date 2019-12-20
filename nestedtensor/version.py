__version__ = '0.0.1.dev201912204+49d9887'
git_version = '49d9887b58b1453e009c0256ba2dac7f1e19ce67'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
