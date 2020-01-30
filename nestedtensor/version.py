__version__ = '0.0.1.dev20201301+ad44b2c'
git_version = 'ad44b2c427453be2d2eccfc051a3c53c63c54686'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
