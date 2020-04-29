__version__ = '0.0.1.dev202042918+a3ec635'
git_version = 'a3ec63582c98dcfd3efcecfd55cf77adc3b03111'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
