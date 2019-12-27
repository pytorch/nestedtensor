__version__ = '0.0.1.dev2019122721+b644f10'
git_version = 'b644f1000123f2a142aa3076c32967957fd72aa1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
