__version__ = '0.0.1.dev20206104+379f666'
git_version = '379f6666875314af0e3afd9aa1bb1adebe9679b1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
