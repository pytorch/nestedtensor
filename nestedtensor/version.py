__version__ = '0.0.1.dev20208251+17bf81d'
git_version = '17bf81da887e618d3931b229795b6320b26b3f78'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
