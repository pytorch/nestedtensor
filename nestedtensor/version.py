__version__ = '0.0.1.dev202083018+8b0cde3'
git_version = '8b0cde37872046ddb7082776519343a3d19bcc42'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
