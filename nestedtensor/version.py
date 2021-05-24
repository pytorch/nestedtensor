__version__ = '0.1.4+4502a48'
git_version = '4502a4832a4e476ede38e746a8039e079ab2018d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
