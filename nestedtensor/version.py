__version__ = '0.0.1.dev20201132+4c4d8fe'
git_version = '4c4d8fedbfced7cec69c18f9b8c06fd04f2d2649'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
