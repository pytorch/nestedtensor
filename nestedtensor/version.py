__version__ = '0.1.4+868ecfd'
git_version = '868ecfd88eaf895c65ae37b16cc3c6896a26f148'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
