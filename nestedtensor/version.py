__version__ = '0.0.1.dev20208254+7598611'
git_version = '759861199cb721bbd76fd56d787881ac096629f6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
