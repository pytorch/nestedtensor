__version__ = '0.0.1.dev2019121620+2a4f506'
git_version = '2a4f5063aab5f54246e1362afc636597ede51868'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
