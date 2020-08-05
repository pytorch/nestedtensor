__version__ = '0.0.1.dev20208514+23cd5f4'
git_version = '23cd5f4421ef6ebb4748211ae866694f0c668828'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
