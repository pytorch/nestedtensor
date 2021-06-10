__version__ = '0.1.4+37bb751'
git_version = '37bb75127289d4aac113200126c1ea05dc68c74f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
