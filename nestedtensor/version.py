__version__ = '0.0.1.dev202092921+5b2ca09'
git_version = '5b2ca090a7a338e915f1b0706beb843191d3830f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
