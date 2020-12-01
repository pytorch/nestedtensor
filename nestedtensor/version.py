__version__ = '0.0.1.dev20201210+f853a68'
git_version = 'f853a68f343abfec7f7cb411c0e14ea23dd7e061'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
