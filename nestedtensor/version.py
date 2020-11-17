__version__ = '0.0.1.dev202011172+dc663e7'
git_version = 'dc663e71c7fc013b1533adf15ebd8dfa74380e64'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
