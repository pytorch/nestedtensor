__version__ = '0.0.1.dev20208254+e8c591a'
git_version = 'e8c591a13f9634f64d41241cd7770c5ce1fd9523'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
