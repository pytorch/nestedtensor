__version__ = '0.0.1.dev202092217+644f4ab'
git_version = '644f4abe4572a86bb3f6f46698c6c179dff2a915'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
