__version__ = '0.0.1.dev20202420+bc35af3'
git_version = 'bc35af313790797a9217164306f8a12c5195db06'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
