__version__ = '0.0.1.dev2019121320+a90ada4'
git_version = 'a90ada43b22e65be46ba1a93cd70c4b6d14cbfeb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
