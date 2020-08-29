__version__ = '0.0.1.dev20208295+435e769'
git_version = '435e76993f6898461e97402b737986b7f47023d3'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
