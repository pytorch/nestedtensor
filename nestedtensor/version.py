__version__ = '0.0.1.dev2020112416+560d64e'
git_version = '560d64ebd36443040b9f4de58e28a685663b2bfd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
