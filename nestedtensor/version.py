__version__ = '0.0.1.dev201912202+d610758'
git_version = 'd610758a184dc9bf59e4f29e9a7ad75ed670b1ab'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
