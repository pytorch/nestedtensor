__version__ = '0.0.1.dev2020102823+d774437'
git_version = 'd7744375d86adc57d20993371c29ddb2661d5a8c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
